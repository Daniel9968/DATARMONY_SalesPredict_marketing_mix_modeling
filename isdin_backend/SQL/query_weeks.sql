SELECT 
    CONCAT('week_', week) AS week 
FROM (
    SELECT DISTINCT 
        EXTRACT(ISOWEEK FROM date) AS week 
    FROM `prj-dt-pro-dwh-dmt-data.ds_fusion_marketing_mix_modeling.tbl-dmt-fusion-marketing_mix_modeling-sellout_simplified`
    WHERE EXTRACT(ISOYEAR FROM date) > 2020 ORDER BY 1) 
WHERE week != 1