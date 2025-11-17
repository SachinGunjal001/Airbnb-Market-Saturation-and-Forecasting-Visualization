-- 🔗 Join Airbnb listings with external zoning data to enrich listings with zone-level info
-- This helps associate each listing with its designated land-use classification (e.g., Residential, Commercial)
-- Useful for urban planning, understanding regulation impact, and spatial segmentation of Airbnb growth

SELECT * FROM [dbo].[AB_NYC]

-- 🧾 Business Use Case:
-- Allows us to analyze trends like: "Are most listings operating in residential zones?"
-- Helps regulators or city officials understand zoning compliance and saturation by area

--------------------------------------------------------------------------------------------

-- 💰 Aggregate pricing data to evaluate borough-wise Airbnb performance

SELECT 
    neighbourhood_group,              -- Borough-level grouping (e.g., Manhattan, Brooklyn, Bronx)
    AVG(price) AS avg_price,          -- Calculates average listing price in each borough
    COUNT(*) AS total_listings        -- Counts total number of listings per borough
FROM 
      [dbo].[AB_NYC]                -- Source: cleaned and filtered listing data
GROUP BY 
    neighbourhood_group;              -- Grouping condition to perform borough-level aggregation

-- 🧾 Business Use Case:
-- Provides KPIs for dashboards — which boroughs are most expensive or most saturated?
-- Informs investment decisions, marketing targeting, and policy intervention areas
