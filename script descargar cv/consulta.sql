select ap.user_ptr_id, ap.gender , ap.birth_date , ap.disability_type , gg.score , gt.trait, gj.job_application
from accounts_businessuser ab, gamesandtests_jobapplication gj, gamesandtests_userjobapplication gu, accounts_personaluser ap,
accounts_user au, gamesandtests_gameresponsetrait gg, genomes_trait gt
where gj.business_user_id_id = ab.user_ptr_id and gu.job_application_id_id = gj.id
and ap.user_ptr_id = gu.personal_user_id_id  and au.id = ap.user_ptr_id and gg.user_id_id = ap.user_ptr_id and gt.id = gg.trait_id_id and ab.business_user like 'Copeval' 


SELECT accounts_personaluser.user_ptr_id as user_id, cv
    FROM accounts_personaluser
    WHERE cv is not NULL-- NOT NULL CV
    AND cv <> ''
    AND cv LIKE '%.pdf'

https://genoma-archives.s3.us-east-2.amazonaws.com/
