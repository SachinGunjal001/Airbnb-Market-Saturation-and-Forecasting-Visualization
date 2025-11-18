
from urllib.parse import quote_plus
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
import datetime
import math

warnings.simplefilter('ignore', ConvergenceWarning)
st.set_page_config(page_title="Airbnb NYC — Enterprise Dashboard", layout="wide", page_icon="🏙️")

# ---------------------------
# Configuration: SQL Server
# ---------------------------
SQL_SERVER = r"SACHIN\SQLEXPRESS"
SQL_DATABASE = "Sachin"
DRIVER = "ODBC Driver 17 for SQL Server"

quoted_driver = quote_plus(DRIVER)
SQLALCHEMY_CONN_STR = f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DATABASE}?driver={quoted_driver}&trusted_connection=yes"
# Create engine lazily in SQLManager

# ---------------------------
# Utility functions
# ---------------------------
def safe_to_int(x, default=0):
    try:
        return int(x)
    except:
        return default

def safe_to_float(x, default=np.nan):
    try:
        return float(x)
    except:
        return default

def format_currency(x):
    try:
        return f"₹{int(x):,}"
    except:
        return x

def month_number_to_name(m):
    try:
        return datetime.datetime(1900, int(m), 1).strftime('%b')
    except:
        return str(m)

# ---------------------------
# Data Loader & SQL Manager
# ---------------------------
class DataLoader:
    def __init__(self, csv_path="AB_NYC_2019_cleaned.csv"):
        self.csv_path = csv_path
        self.raw = None

    def load_csv(self):
        df = pd.read_csv(self.csv_path)
        self.raw = df.copy()
        return self.raw

class SQLManager:
    def __init__(self, conn_str=SQLALCHEMY_CONN_STR, table_name="ab_nyc_listings"):
        self.conn_str = conn_str
        self.table_name = table_name
        self.engine = None

    def get_engine(self):
        if self.engine is None:
            self.engine = create_engine(self.conn_str, fast_executemany=True)
        return self.engine

    def write_df_to_sql(self, df, if_exists='replace'):
        eng = self.get_engine()
        df.to_sql(self.table_name, con=eng, index=False, if_exists=if_exists)
        return True

    def read_sql(self, query):
        eng = self.get_engine()
        return pd.read_sql(query, con=eng)

    def table_exists(self):
        eng = self.get_engine()
        with eng.connect() as conn:
            res = conn.execute(text(f"SELECT OBJECT_ID('{self.table_name}', 'U')")).scalar()
            return res is not None

# ---------------------------
# Preprocessing + Feature Engineering
# ---------------------------
class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def basic_clean(self):
        df = self.df
        # Standardize column names
        df.columns = [c.strip() for c in df.columns]

        # Basic dtypes
        if 'last_review' in df.columns:
            try:
                df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce', dayfirst=False)
            except:
                df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

        # Fill numeric nulls
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Fill small set of missing numeric values with 0 or median
        for c in ['reviews_per_month', 'number_of_reviews', 'price', 'availability_365', 'minimum_nights']:
            if c in df.columns:
                df[c] = df[c].fillna(0)

        # Name cleaning
        if 'name' in df.columns:
            df['name'] = df['name'].fillna('No Title')

        # Host name
        if 'host_name' in df.columns:
            df['host_name'] = df['host_name'].fillna('Unknown Host')

        self.df = df
        return self

    def add_features(self):
        df = self.df

        # price per person (if accommodates exists)
        if 'price' in df.columns and 'minimum_nights' in df.columns:
            df['price_per_min_night'] = df['price'] / df['minimum_nights'].replace(0, 1)
        if 'price' in df.columns and 'month' in df.columns:
            # keep provided price_per_min_night if exists in file; else compute fallback
            if 'price_per_min_night' not in df.columns:
                df['price_per_min_night'] = df['price'] / df['minimum_nights'].replace(0, 1)

        # price bins
        if 'price' in df.columns:
            bins = [-1, 50, 100, 200, 400, 800, df['price'].max()+1]
            labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Luxury']
            labels = [f"{bins[i]} - {bins[i+1]}" for i in range(len(bins)-1)]
            df['price_segment'] = pd.cut(df['price'], bins=bins, labels=labels, include_lowest=True)


        # room binary
        if 'room_type' in df.columns:
            df['is_entire_home'] = df['room_type'].apply(lambda x: 1 if str(x).lower().find('entire') >= 0 else 0)

        # borough mapping
        if 'neighbourhood_group' in df.columns:
            df['borough'] = df['neighbourhood_group']

        # month name (if numeric month column exists)
        if 'month' in df.columns:
            df['month_name'] = df['month'].apply(lambda x: month_number_to_name(x))

        # reviews activity: days since last review
        if 'last_review' in df.columns:
            df['days_since_last_review'] = (pd.Timestamp('today') - df['last_review']).dt.days.fillna(9999).astype(int)

        # revenue proxy: price * availability fraction
        if 'price' in df.columns and 'availability_365' in df.columns:
            df['availability_rate'] = df['availability_365'] / 365.0
            df['revenue_proxy'] = df['price'] * df['availability_rate'] * 12  # annualized rough proxy

        # coordinates check
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['lat_lon_ok'] = df[['latitude','longitude']].apply(lambda r: not (pd.isna(r['latitude']) or pd.isna(r['longitude'])), axis=1)

        # fill any remaining categorical nulls
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        for c in cat_cols:
            df[c] = df[c].fillna('Not Specified')

        self.df = df
        return self

    def final(self):
        # Final cleaning touches
        df = self.df
        # Deduplicate by id if present
        if 'id' in df.columns:
            df = df.drop_duplicates(subset=['id'])
        # Force types for key columns
        for c in ['price','minimum_nights','number_of_reviews','availability_365']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
        self.df = df
        return df

# ---------------------------
# Visualizations: grouped helpers
# ---------------------------
class Viz:
    @staticmethod
    def kpi_row(kpis):
        cols = st.columns(len(kpis))
        for c, (label, value, delta) in zip(cols, kpis):
            if delta is None:
                c.metric(label=label, value=value)
            else:
                c.metric(label=label, value=value, delta=delta)

    @staticmethod
    def top_table(df, n=10, title=None):
        if title:
            st.markdown(f"#### {title}")
        st.dataframe(df.head(n), use_container_width=True)

    @staticmethod
    def plot_price_histogram(df):
        fig = px.histogram(df, x='price', nbins=40, title='Price Distribution', marginal='box')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def room_type_pie(df):
        if 'room_type' not in df.columns:
            return
        agg = df['room_type'].value_counts().reset_index()
        agg.columns = ['room_type','count']
        fig = px.pie(agg, names='room_type', values='count', title='Room Type Mix', hole=0.45)
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def borough_bar(df):
        if 'borough' not in df.columns:
            return
        agg = df.groupby('borough')['id'].count().reset_index().sort_values('id', ascending=False)
        agg.columns = ['borough','listings']
        fig = px.bar(agg, x='borough', y='listings', title='Listings by Borough', text='listings')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def price_box_by_borough(df):
        if 'borough' not in df.columns:
            return
        fig = px.box(df, x='borough', y='price', points='all', title='Price distribution by Borough')
        st.plotly_chart(fig, use_container_width=True, height=500)

    @staticmethod
    def price_time_trend(df):
        if 'month' not in df.columns:
            return
        agg = df.groupby('month')[['price']].mean().reset_index().sort_values('month')
        agg['month_name'] = agg['month'].apply(month_number_to_name)
        fig = px.line(agg, x='month_name', y='price', title='Average Price by Month', markers=True)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def top_hosts_table(df, top_n=10):
        if 'host_name' not in df.columns:
            return
        agg = df.groupby('host_name')['id'].count().reset_index().sort_values('id', ascending=False).head(top_n)
        agg.columns = ['host_name','listings']
        st.dataframe(agg, use_container_width=True)

    @staticmethod
    def scatter_map(df, zoom=10):
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return
        fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='price_segment',
                                size='price', hover_name='name', hover_data=['price','room_type','neighbourhood'],
                                title='Listings Map (clustered by price segment)', zoom=10, height=600)
        fig.update_layout(mapbox_style='open-street-map', margin={"r":0,"t":35,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def correlation_heatmap(df):
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            return
        corr = num.corr()
        fig = px.imshow(corr, text_auto=True, title='Correlation Matrix (numeric features)')
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def availability_gauge(df):
        if 'availability_365' not in df.columns:
            return
        avg_avail = df['availability_365'].mean()
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_avail,
            title = {'text': "Avg Availability (days/year)"},
            gauge = {'axis': {'range': [0, 365]}}
        ))
        st.plotly_chart(fig, use_container_width=True, height=300)

    @staticmethod
    def reviews_scatter(df):
        if 'reviews_per_month' not in df.columns:
            return
        fig = px.scatter(df, x='reviews_per_month', y='price', color='room_type', size='number_of_reviews',
                         hover_data=['name','host_name'], title='Price vs Reviews per Month')
        st.plotly_chart(fig, use_container_width=True, height=450)

    @staticmethod
    def cluster_and_show(df, n_clusters=6):
        # Select features
        if not {'latitude','longitude','price'}.issubset(df.columns):
            return
        data = df[['latitude','longitude','price']].dropna()
        scaler = StandardScaler()
        X = scaler.fit_transform(data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        data = data.copy()
        data['cluster'] = clusters
        # Plot clusters on map
        fig = px.scatter_mapbox(data, lat='latitude', lon='longitude', color='cluster', size='price',
                                hover_data=['price'], title=f'Geospatial Clusters (k={n_clusters})', zoom=10)
        fig.update_layout(mapbox_style='open-street-map', margin={"r":0,"t":35,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True, height=600)
        # cluster price summary
        summary = data.groupby('cluster')['price'].agg(['count','mean','median']).reset_index()
        st.dataframe(summary, use_container_width=True)

    @staticmethod
    def price_distribution_by_neighbourhood(df, limit=12):
        if 'neighbourhood' not in df.columns:
            return
        agg = df.groupby('neighbourhood')['price'].median().reset_index().sort_values('price', ascending=False).head(limit)
        fig = px.bar(agg, x='neighbourhood', y='price', title=f'Top {limit} neighbourhoods by median price', text='price')
        st.plotly_chart(fig, use_container_width=True, height=450)

    @staticmethod
    def boxplot_price_by_room(df):
        if 'room_type' not in df.columns:
            return
        fig = px.box(df, x='room_type', y='price', title='Price by Room Type (boxplot)')
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def revenue_scatter(df):
        if 'revenue_proxy' not in df.columns:
            return
        fig = px.scatter(df, x='availability_365', y='revenue_proxy', color='price_segment',
                         hover_data=['price','name','host_name'], title='Revenue proxy vs Availability')
        st.plotly_chart(fig, use_container_width=True, height=450)

    @staticmethod
    def monthly_heatmap(df):
        if 'month_name' not in df.columns:
            return
        pivot = df.pivot_table(index='neighbourhood', columns='month_name', values='price', aggfunc='median', fill_value=0)
        # Limit for display
        if pivot.shape[0] > 30:
            pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=False).head(30)
        fig = px.imshow(pivot, title='Median Price Heatmap (neighbourhood x month)')
        st.plotly_chart(fig, use_container_width=True, height=700)

    @staticmethod
    def price_vs_minimum_nights(df):
        if 'minimum_nights' not in df.columns:
            return
        fig = px.scatter(df, x='minimum_nights', y='price', color='room_type', hover_data=['name'], title='Price vs Minimum Nights')
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def top_neighbourhoods_by_listings(df, top_n=15):
        agg = df['neighbourhood'].value_counts().reset_index().rename(columns={'index':'neighbourhood','neighbourhood':'count'}).head(top_n)
        fig = px.bar(agg, x='neighbourhood', y='count', title=f'Top {top_n} Neighbourhoods by # Listings', text='count')
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Forecasting Helpers
# ---------------------------
def forecast_price_monthly(df, periods=6):
    """
    Simple Holt-Winters seasonal forecast on monthly average price.
    Returns dataframe with historical and forecast.
    """
    if 'month' not in df.columns or 'price' not in df.columns:
        return None

    # monthly avg by month number (1..12)
    monthly = df.groupby('month')['price'].mean().reset_index().sort_values('month')
    # create a time series index - we will assume month cycle; create a faux date for each month for clarity
    monthly = monthly.set_index(pd.to_datetime(monthly['month'].astype(int).astype(str) + '-01-2019'))
    monthly = monthly['price']
    try:
        model = ExponentialSmoothing(monthly, trend='add', seasonal='add', seasonal_periods=12)
        fit = model.fit()
        pred = fit.forecast(periods)
        pred_index = pd.date_range(start=monthly.index.max() + pd.DateOffset(months=1), periods=periods, freq='MS')
        hist = monthly.reset_index().rename(columns={'index':'date','price':'price'})
        forecast_df = pd.DataFrame({'date': pred_index, 'forecast': pred.values})
        return hist, forecast_df
    except Exception as e:
        return None

# ---------------------------
# Main Streamlit App
# ---------------------------
def run_app():
    st.title("Airbnb NYC — Enterprise Storytelling Dashboard")
    st.markdown("This dashboard loads the provided CSV, writes to SQL Server, and presents a large set of analytics, visuals and storytelling modules. Developed for exploratory and business reporting use-cases.")

    # Left sidebar - actions and navigation
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/gopiashokan/Airbnb-Analysis/main/airbnb_banner.jpg")
        menu = st.radio("Navigation", (
            "Overview",
            "Price Insights",
            "Neighbourhood Explorer",
            "Host Analysis",
            "Time Series & Forecast",
            "Clustering & Maps",
            "Correlations & Heatmaps",
            "Revenue Estimator",
            "Data & SQL"
        ))
        st.write("---")
        st.write("SQL Server")
        st.text(SQL_SERVER)
        st.text(SQL_DATABASE)
        migrate = st.button("Write to SQL Server (replace table)")
        if migrate:
            st.info("Attempting to write to SQL Server...")

    # Load data
    loader = DataLoader()
    df_raw = loader.load_csv()
    st.sidebar.markdown(f"Rows: {df_raw.shape[0]}  Columns: {df_raw.shape[1]}")

    # Preprocess
    prep = Preprocessor(df_raw)
    prep.basic_clean().add_features()
    df = prep.final()

    # Build SQL manager
    sqlm = SQLManager()

    # If migrate requested - write to SQL
    if migrate:
        try:
            sqlm.write_df_to_sql(df, if_exists='replace')
            st.success(f"Data written to SQL table `{sqlm.table_name}` in database {SQL_DATABASE}")
        except Exception as e:
            st.error("Failed writing to SQL Server: " + str(e))

    # Overview Page
    if menu == "Overview":
        st.header("Executive Overview — Key Business Insights")
        # KPIs
        total_listings = df.shape[0]
        avg_price = df['price'].mean() if 'price' in df.columns else 0
        median_price = df['price'].median() if 'price' in df.columns else 0
        avg_avail = df['availability_365'].mean() if 'availability_365' in df.columns else 0
        top_borough = df['borough'].mode().iloc[0] if 'borough' in df.columns else 'N/A'
        kpis = [
            ("Total Listings", f"{total_listings:,}", None),
            ("Avg Price", f"${avg_price:.0f}", f"{(avg_price-median_price):+.0f}"),
            ("Median Price", f"${median_price:.0f}", None),
            ("Avg Availability (days)", f"{avg_avail:.1f}", None),
            ("Top Borough", top_borough, None)
        ]
        Viz.kpi_row(kpis)
        st.markdown("### Quick Visuals")
        col1, col2 = st.columns(2)
        with col1:
            Viz.plot_price_histogram(df)
            Viz.room_type_pie(df)
        with col2:
            Viz.borough_bar(df)
            Viz.availability_gauge(df)

        st.markdown("### Snapshot of Top Listings")
        Viz.top_table(df[['id','name','host_name','neighbourhood','borough','room_type','price','minimum_nights']].sort_values('price', ascending=False), n=10)

        st.markdown("### Short narrative")
        st.write("""**Topline:** Median price is telling — focus required on 'Very High' & 'Luxury' segments. Neighborhood-level price variance is high, indicating an opportunity for targeted yield management and borough-specific promotions.
                **Actionable:** Build dynamic pricing for high-variance neighbourhoods, and prioritize hosts in high-revenue-proxy segments for partnerships.""")

    # Price Insights Page
    elif menu == "Price Insights":
        st.header("Price Insights & Distribution")
        col1, col2 = st.columns([2,1])
        with col1:
            Viz.plot_price_histogram(df)
            Viz.price_distribution_by_neighbourhood(df, limit=20)
            Viz.price_time_trend(df)
        with col2:
            st.markdown("#### Price Segments")
            st.dataframe(df.groupby('price_segment')['id'].count().reset_index().rename(columns={'id':'count'}).sort_values('count', ascending=False), use_container_width=True)
            st.markdown("#### Boxplot: price by room type")
            Viz.boxplot_price_by_room(df)
            st.markdown("#### Price vs Minimum Nights")
            Viz.price_vs_minimum_nights(df)

    # Neighbourhood Explorer
    elif menu == "Neighbourhood Explorer":
        st.header("Neighbourhood Explorer — drill down and tell the story")
        left, right = st.columns([2,1])
        with right:
            boroughs = ['All'] + sorted(df['borough'].unique().tolist()) if 'borough' in df.columns else ['All']
            chosen_borough = st.selectbox("Select Borough", boroughs)
            nbhd_list = ['All']
            if chosen_borough != 'All':
                nbhd_list = ['All'] + sorted(df[df['borough']==chosen_borough]['neighbourhood'].unique().tolist())
            nbhd = st.selectbox("Select Neighbourhood", nbhd_list)
            min_price, max_price = int(df['price'].min()), int(df['price'].max())
            price_range = st.slider("Price Range", min_value=min_price, max_value=max_price, value=(min_price, max_price))
            room_types = ['All'] + sorted(df['room_type'].unique().tolist()) if 'room_type' in df.columns else ['All']
            chosen_room = st.selectbox("Room Type", room_types)
            show_top = st.slider("Show Top N Listings (by price)", 5, 50, 10)

        # Filter
        filt = df.copy()
        if chosen_borough != 'All':
            filt = filt[filt['borough']==chosen_borough]
        if nbhd != 'All':
            filt = filt[filt['neighbourhood']==nbhd]
        if chosen_room != 'All':
            filt = filt[filt['room_type']==chosen_room]
        filt = filt[(filt['price']>=price_range[0]) & (filt['price']<=price_range[1])]

        st.markdown(f"#### Listings (filtered): {filt.shape[0]} rows")
        Viz.scatter_map(filt)
        st.markdown("#### Top listing snapshot")
        Viz.top_table(filt.sort_values('price', ascending=False)[['id','name','host_name','neighbourhood','price','room_type','availability_365']], n=show_top)

        st.markdown("#### Neighbourhood storytelling")
        st.write("""
        - Compare median price vs availability to find underpriced high-availability areas.
        - Identify hosts with multiple listings for potential co-marketing.
        - Use price_segment to speed up revenue-management rules.
        """)

    # Host Analysis
    elif menu == "Host Analysis":
        st.header("Host Analysis — behaviour & concentration")
        col1, col2 = st.columns([2,1])
        with col1:
            Viz.top_hosts_table(df, top_n=20)
            st.markdown("### Host Listings Map")
            host_counts = df.groupby(['host_name','latitude','longitude']).size().reset_index(name='count')
            host_counts = host_counts[host_counts['count']>1]
            if not host_counts.empty:
                fig = px.scatter_mapbox(host_counts, lat='latitude', lon='longitude', size='count', hover_name='host_name', title='Hosts with multiple listings', zoom=10)
                fig.update_layout(mapbox_style='open-street-map')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hosts with multiple listings found in this small sample.")

        with col2:
            st.markdown("### Host metrics")
            top_host = df['host_name'].value_counts().idxmax()
            host_kpi = [
                ("Top Host", top_host, None),
                ("Host Listing Count", f"{df['host_name'].value_counts().max()}", None),
                ("Avg Price per Host", f"${df.groupby('host_name')['price'].mean().mean():.0f}", None)
            ]
            Viz.kpi_row(host_kpi)

        st.markdown("### Host-level Price dispersion")
        host_price = df.groupby('host_name')['price'].agg(['count','mean','median']).sort_values('count', ascending=False).head(30).reset_index()
        st.dataframe(host_price, use_container_width=True)

    # Time Series & Forecast
    elif menu == "Time Series & Forecast":
        st.header("Time Series & Forecasting")
        st.markdown("We use a simple Holt-Winters model on monthly average price (seasonal period = 12).")
        result = forecast_price_monthly(df, periods=6)
        if result is None:
            st.info("Insufficient monthly data for forecasting in this dataset.")
        else:
            hist, forecast_df = result
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist['date'], y=hist['price'], mode='lines+markers', name='Historical Avg Price'))
            fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['forecast'], mode='lines+markers', name='Forecast'))
            fig.update_layout(title="Monthly Average Price — Historical + Forecast", xaxis_title="Month", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Seasonality: monthly boxplots")
        if 'month_name' in df.columns:
            fig = px.box(df, x='month_name', y='price', title='Price by Month (boxplot)')
            st.plotly_chart(fig, use_container_width=True)

    # Clustering & Maps
    elif menu == "Clustering & Maps":
        st.header("Clustering and Geospatial Storytelling")
        st.markdown("KMeans clustering on (lat, lon, price). Use clusters to define micro-markets.")
        n_clusters = st.slider("Choose K for KMeans", 2, 12, 6)
        Viz.cluster_and_show(df, n_clusters=n_clusters)

        st.markdown("### Map filters + heatmap")
        Viz.scatter_map(df)
        st.markdown("### Neighborhood monthly heatmap (top neighbourhoods)")
        Viz.monthly_heatmap(df)

    # Correlations & Heatmaps
    elif menu == "Correlations & Heatmaps":
        st.header("Correlations & Feature Relationships")
        Viz.correlation_heatmap(df)
        st.markdown("### Price vs Reviews vs Availability")
        cols = st.columns(3)
        with cols[0]:
            st.plotly_chart(px.histogram(df, x='reviews_per_month', nbins=30, title='Reviews per Month distribution'), use_container_width=True)
        with cols[1]:
            st.plotly_chart(px.histogram(df, x='availability_365', nbins=30, title='Availability distribution'), use_container_width=True)
        with cols[2]:
            st.plotly_chart(px.histogram(df, x='minimum_nights', nbins=30, title='Minimum nights distribution'), use_container_width=True)

    # Revenue Estimator
    elif menu == "Revenue Estimator":
        st.header("Revenue Estimator — quick scenario builder")
        st.markdown("Use price, availability and occupancy rate assumptions to estimate annual revenue per listing.")
        sample = df.sample(20) if df.shape[0] > 20 else df.copy()
        chosen_id = st.selectbox("Choose Listing (id)", sample['id'].tolist())
        listing = df[df['id']==chosen_id].iloc[0]
        st.markdown(f"**Selected listing:** {listing.get('name','-')}, Host: {listing.get('host_name','-')}, Borough: {listing.get('borough','-')}")
        base_price = float(listing.get('price', 0))
        avail_days = int(listing.get('availability_365', 0))
        occupancy = st.slider("Assumed occupancy rate (%)", 10, 100, 50)
        avg_nights_stayed_per_booking = st.slider("Avg nights per booking", 1, 14, int(listing.get('minimum_nights',1)))
        monthly_rev = (base_price * (occupancy/100.0) * (avail_days/365.0)) * (30/avg_nights_stayed_per_booking)
        annual_rev = monthly_rev * 12
        st.metric("Estimated Monthly Revenue", f"${monthly_rev:,.0f}")
        st.metric("Estimated Annual Revenue", f"${annual_rev:,.0f}")

        st.markdown("### Portfolio revenue by price segment")
        seg = df.groupby('price_segment')['revenue_proxy'].sum().reset_index()
        st.plotly_chart(px.bar(seg, x='price_segment', y='revenue_proxy', title='Estimated Revenue (proxy) by Price Segment', text='revenue_proxy'), use_container_width=True)

    # Data & SQL operations
    elif menu == "Data & SQL":
        st.header("Data & SQL Operations")
        st.markdown("### Raw data preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.markdown("### Download cleaned CSV")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download cleaned CSV", csv, "ab_nyc_cleaned_for_sql.csv", "text/csv")

        st.markdown("### SQL Server actions")
        st.write(f"Table target: `{sqlm.table_name}`")
        if sqlm.table_exists():
            st.success("Table exists in SQL Server (may be overwritten).")
        else:
            st.info("Table does not exist yet on SQL Server.")
        if st.button("Write to SQL Server (append)"):
            try:
                sqlm.write_df_to_sql(df, if_exists='append')
                st.success("Appended to SQL Server table.")
            except Exception as e:
                st.error("Failed to append: " + str(e))

        st.markdown("### Sample SQL read (10 rows)")
        try:
            q = f"SELECT TOP 10 * FROM {sqlm.table_name}"
            sample_sql = sqlm.read_sql(q)
            st.dataframe(sample_sql, use_container_width=True)
        except Exception as e:
            st.info("Unable to read from SQL Server: " + str(e))

    # Footer / credits
    st.write("---")
    st.caption("Built for exploratory analysis and storytelling. Adjust visuals and filters to fit production requirements. For large-scale production: move ETL out of Streamlit, use stored procedures, and schedule forecasting jobs.")

# Run
if __name__ == "__main__":
    run_app()
