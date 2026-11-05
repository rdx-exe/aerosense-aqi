---
name: dataset-source
description: Historical AQI dataset research and collection strategy
metadata:
  type: project
---

The OpenAQ API v2 is decommissioned (410 Gone). Accessing the v3 API requires an API key, which is not ideal for an open-source project. 

**Plan**: 
1. Use the Kaggle/GitHub-accessible `city_day.csv` (CPCB India data) as the primary historical source, as it covers 2015-2020.
2. For 2021-present, manually augment with data from the [CPCB official website](https://cpcb.nic.in/) or [AQICN](https://aqicn.org/data-platform/register/).
3. Acknowledge that the dataset is a composite and document the merge strategy in the `data/` folder audit reports.

**Why:** Reliable, historical, and free to download without API keys.
**How to apply:** Use the `src.scripts.download_data` as a template but switch to local CSV management. Update `README.md` to reflect this change.
[[project-structure]]
[[data-pipeline]]
---