SELECT COALESCE(SUM(spend), 0) AS "spend!"
FROM "LiteLLM_SpendLogs"
WHERE api_key = $1
  AND "startTime" >= $2
