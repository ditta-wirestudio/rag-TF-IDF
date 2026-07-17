# API Documentation

The REST API is rate limited to **100 requests per minute** on the free tier and
1,000 requests per minute on Pro. Exceeding the limit returns HTTP 429. Auth uses
a Bearer token in the `Authorization` header.
