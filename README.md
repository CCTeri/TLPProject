# TLPProject: Product Share Model
This product is a lightweight analytics engine that, for any given origin–destination route and time period, 
computes how each cargo “product” (General, Perishables, Pharma, DGR, etc.) shares the total market volume.

*In short, this tool is to provide the product type shares per O&D.*

This works in:
- Ingests monthly O→D shipment data
- Aggregates by product and route to calculate each product’s weight share
- Builds time-series features (lags, rolling averages, seasonality flags)
- Trains a single global regression model to forecast next-month shares
- Outputs, per route, the product most likely to lead and its projected percentage of that market
- Side feature: product stability indicator

By surfacing which products dominate which corridors and where demand is shifting, it supports pricing, capacity planning, performance monitoring and risk management in a single, end-to-end pipeline.


## Questions that can be answered
- Which product are strong at this specific market?
- Where should we deploy new service offerings or capacity based on forecasted share growth?
- Which market is rising a demand in this specific type of product and how quickly are they growing?
- Which routes are the most “concentrated” in a single product versus those with diversified mixes?
- How do product shares correlate with average yield or revenue share?
- Which products exhibit the strongest seasonal patterns on specific origin-destination pairs?


## Point of this product: 
by looking at the data, companies can see which markets having what type of products.

- **Market Demand Monitoring**
  - It helps pricing adjustment by the service the company offer and compare with the competitions and the market demand.
  
- **Capacity Planning**
  - Capacity can be adjusted by the service the company can offer and based on the demand from the market.
  
- **Performance Monitoring**
  - Realize rising markets that the company can offer more capacity based on the demand trend from the past.

- **Risk Management**
  - Prevent any declining market for a specific type of product and in case the company offering the type of service.
  
- **Marketing Strategy**
  - Apply typer-targeted promotions to rising market and understand brand positioning
  


            


