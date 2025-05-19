WITH

  -- Tabla de unidades
  sellout AS (
    SELECT
      EXTRACT(ISOYEAR FROM date) AS year,
      EXTRACT(MONTH FROM date) AS month,
      EXTRACT(ISOWEEK FROM date) AS week,
      country_id,
      brand,
      bu,
      currency,
      material_id,
      product_id,
      units
    FROM `prj-dt-pro-dwh-dmt-data.ds_fusion_marketing_mix_modeling.tbl-dmt-fusion-marketing_mix_modeling-sellout`
    WHERE
      material_id IS NOT NULL AND
      date <= DATE_SUB(DATE_TRUNC(CURRENT_DATE(), WEEK(SUNDAY)), INTERVAL 1 WEEK) AND
      units > 0 -- Descartamos devoluciones
    ),

  -- Tabla de ventas
  pvpr AS (
    SELECT
      product_id,
      CAST(REPLACE (pvpr, ',', '.') AS FLOAT64) AS pvpr -- Arreglamos el formato
    FROM `prj-dt-pro-dwh-dmt-data.ds_fusion_marketing_mix_modeling.tbl-dmt-fusion-marketing_mix_modeling-product_country_attributes`),

  -- Tabla de campañas
  campaigns AS (
    SELECT
      campaign,
      campaign_alias,
      EXTRACT(ISOYEAR FROM date) AS year,
      EXTRACT(ISOWEEK FROM date) AS week,
      currency,
      spent,
      store,
      platform,
      origen as origin
    FROM `prj-dt-pro-dwh-dmt-data.ds_booster_marketing_mix_modeling.tbl-dmt-booster-marketing_mix_modeling-digital_campaigns`
  ),

  -- Tabla de origen
  campaign_origen AS (SELECT *
  FROM `prj-dt-pro-dwh-dmt-data.ds_booster_marketing_mix_modeling.tbl-dmt-booster-marketing_mix_modeling-buckets`
  WHERE origin != 'connected tv'
  ),

  -- Join de las tablas anteriores
  products_join AS (
    SELECT
      year,
      month,
      week,
      country_id,
      currency,
      bu,
      brand,
      SUM(units) AS units,
      SUM(pvpr.pvpr) AS pvpr
    FROM sellout
    LEFT JOIN pvpr on sellout.product_id = pvpr.product_id
    --WHERE bu = 'Foto' AND -- Seleccionamos aquellos productos de Fotoprotección
    WHERE sellout.product_id IS NOT NULL
    GROUP BY year, month, week, country_id, currency, bu, brand
    ),

  -- Join de las tablas de campaña
  campaign_join AS (
    SELECT
      campaign,
      year,
      week,
      currency,
      spent,
      store,
      platform,
      campaigns.origin,
      campaign_bucket_0
    FROM campaigns
    LEFT JOIN campaign_origen ON campaigns.origin = campaign_origen.origin),


  -- Seleccionamos ventas de España con Fotoprotección con ventas en farmacias
  products_total AS (
    SELECT
      year,
      MIN(month) AS month,
      week,
      country_id,
      currency,
      bu,
      brand,
      SUM(pvpr) AS pvpr,
      SUM(units) AS units
    FROM(
      SELECT
        year,
        month,
        week,
        country_id,
        currency,
        bu,
        brand,
        CAST(SUM(units) AS FLOAT64) AS units,
        SUM(pvpr) AS pvpr
        FROM products_join
      WHERE country_id = {{country}} AND -- Ventas España
      bu = {{bu}} AND brand = {{brand}}
      GROUP BY year, month, week, country_id, currency, bu, brand
      ORDER BY year, month, week) 
    GROUP BY year, week, country_id, currency, bu, brand),

  -- Seleccionamos campañas de España en Fotoprotección
  campaign_aux AS (
    SELECT
      year,
      week,
      campaign_bucket_0 AS campaign_bucket,
      SUM(spent) AS spent
    FROM campaign_join
    WHERE store = LOWER({{country}}) AND REGEXP_CONTAINS(campaign, r{{index_bu}})
    GROUP BY year, week, campaign_bucket
    ORDER BY year, week),

  campaign_total AS (
    SELECT 
      * 
    FROM
    campaign_aux
    PIVOT(SUM(spent) FOR campaign_bucket IN {{list_bucket}})
    campaign_bucket)

--   licon AS (
--     SELECT 
--       CAST(year AS NUMERIC) AS year,
--       CAST(week AS NUMERIC) AS week,
--       CAST(earned_real_acc_kpts AS FLOAT64) AS earned_real_acc_kpts,
--       CAST(earned_real_kpts AS FLOAT64) AS earned_real_kpts,
--       CAST(earned_new_ilrs AS FLOAT64) AS earned_new_ilrs,
--       CAST(earned_ilrs AS FLOAT64) AS earned_ilrs,
--       CAST(redeem_checkout AS FLOAT64) AS redeem_checkout,
--       CAST(redeem_ilrs AS FLOAT64) AS redeem_ilrs
--  FROM `marketing-mix-modeling-386411.aux.ds_df_licon`),

--   promo_cmp AS (
--     SELECT 
--       CAST(year AS NUMERIC) AS year,
--       CAST(week AS NUMERIC) AS week,
--       Other_Price,
--       Other_MA_Price_5Months,
--       Other_Diff_MA_Price_5_Months,
--       Other_Diff_AVG_anual
--  FROM `marketing-mix-modeling-386411.aux.ds_df_promo_cmp`),

--   promo_isd AS (
--     SELECT 
--       CAST(year AS NUMERIC) AS year,
--       CAST(week AS NUMERIC) AS week,
--       ISDIN_Price,
--       ISDIN_MA_Price_5Months,
--       ISDIN_Diff_MA_Price_5_Months,
--       ISDIN_Diff_AVG_anual

--  FROM `marketing-mix-modeling-386411.aux.ds_df_promo_isdin`)

SELECT
*
FROM(
  SELECT
    CASE WHEN products_total.year IS NOT NULL THEN products_total.year ELSE campaign_total.year END year,
    CASE WHEN products_total.week IS NOT NULL THEN products_total.week ELSE campaign_total.week END week,
    * EXCEPT(year, week)
  FROM products_total
  LEFT JOIN campaign_total ON products_total.year = campaign_total.year AND products_total.week = campaign_total.week
  -- LEFT JOIN licon ON products_total.year = licon.year AND products_total.week = licon.week
  -- LEFT JOIN promo_cmp ON products_total.year = promo_cmp.year AND products_total.week = promo_cmp.week
  -- LEFT JOIN promo_isd ON products_total.year = promo_isd.year AND products_total.week = promo_isd.week

  ORDER BY year, week) WHERE year > 2020