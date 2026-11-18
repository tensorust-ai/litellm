SELECT COALESCE(SUM(spend), 0) AS "spend!"
FROM "LiteLLM_SpendLogs"
WHERE "user" = $1
  AND "startTime" >= $2
