# README screenshot placeholders

These files are intentional placeholders. The main `README.md` already points to
each path below, so you only need to overwrite the image file with a real
screenshot — no Markdown edits are required.

## Capture checklist

- Use the dashboard in a desktop browser at **1600 px wide** (or 1920 px).
- Keep the **light theme** and leave the **sidebar visible** for desktop shots.
- Expand the relevant tab or panel before capturing.
- Use PNG, full colour, and avoid browser chrome where possible.
- Do not include personal data, API keys, or unrelated tabs.
- For the mobile shot, use a **390 × 844** viewport and collapse the sidebar.

## Files and prompts

| File | Section / state to capture | Suggested size |
| :--- | :--- | :--- |
| `dashboard-overview.png` | **Overview** — hero, KPI cards, 120-day AQI trend, project/data cards | 1600 × 1000 |
| `dashboard-historical.png` | **Historical air quality** → **Pollutant analysis** — scatter, correlation matrix, summary table | 1600 × 1200 |
| `dashboard-prediction.png` | **AQI prediction** — inputs, model-generated forecast card, gauge, origin/target dates | 1600 × 1100 |
| `dashboard-performance.png` | **Model performance** → **Predictions & residuals** — metric cards, scatter, residual plot | 1600 × 1100 |
| `dashboard-insights.png` | **Environmental insights** — KPI cards, key findings, high-pollution timeline | 1600 × 1200 |
| `dashboard-project-info.png` | **Project information** — provenance, model contract, integrity, limitations | 1600 × 1100 |
| `dashboard-mobile.png` | **Overview** at a 390 px viewport — responsive layout demonstration | 390 × 844 |

## Quick capture with Playwright (optional)

If you want repeatable screenshots, install Playwright and use a script like:

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto("http://localhost:8501", wait_until="networkidle")
    page.wait_for_selector("text=Kolkata Air Quality Intelligence")
    page.wait_for_timeout(1500)
    page.screenshot(path="docs/images/screenshots/dashboard-overview.png", full_page=True)
    browser.close()
```

Repeat for each section by clicking the matching sidebar item, then replace the
corresponding PNG. The placeholders themselves are not used by the app.
