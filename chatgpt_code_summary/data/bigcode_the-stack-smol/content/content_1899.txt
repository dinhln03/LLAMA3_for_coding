class SqlDataQualityQueries:
    establisment_company_relation_check = ("""
    -- looks for registration without relation with compay
    -- for have a database with full information needs to return total equal to zero (establisment + company)
    SELECT count(e.basiccnpj) as total_without_relation from open_data.fact_establishment e
    LEFT JOIN open_data.dim_company c ON c.basiccnpj = e.basiccnpj 
    WHERE c.basiccnpj is null;
    """, "== 1", "== 0")    