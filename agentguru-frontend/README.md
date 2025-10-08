I have created the markdown file **`AGENTS_CORE_LOGIC.md`** for you, which includes the data flow overview, the weighted signal definitions, and the mathematical formulas for the Prospect Score calculation.

---

# AGENT GURU: CORE PROSPECTING LOGIC

This document outlines the core logic of the Advanced Prospecting Agent, specifically detailing the Signal Engine rules and the proprietary Prospect Score calculation.

## 1. Data Processing Flow

The system executes a two-step process to generate an actionable lead:

1.  **Research (`/research` endpoint):** The agent fetches raw data from three sources: **Apollo** (Company/Contact info), **Serper** (General News), and **AlphaVantage** (Financial News & Sentiment). All news articles are merged and saved to MongoDB.
2.  **Analysis (`/signals` endpoint):** The raw articles are analyzed by the Signal Engine to detect keywords, assign a score, and generate a priority recommendation.

---

## 2. Signal Engine: Weight Definitions

The Signal Engine detects key events by matching keywords against the combined article content (title, snippet, summary). Each detected signal contributes a specific, pre-assigned **weight** to the total Prospect Score.

| Signal Type           | Keywords (Examples)                                            | Weight (Points) | Buying Intent |
| :-------------------- | :------------------------------------------------------------- | :-------------- | :------------ |
| **funding**           | `funding`, `raised`, `investment`, `series a`, `seed round`    | **25**          | High          |
| **acquisition**       | `acquisition`, `acquires`, `merger`, `takeover`, `M&A`         | **25**          | High          |
| **leadership_change** | `ceo`, `cto`, `appointed`, `new chief`, `executive`            | **22**          | High          |
| **revenue_growth**    | `revenue`, `profit`, `earnings`, `growth`, `record`            | **20**          | Medium-High   |
| **expansion**         | `expansion`, `expanding`, `new office`, `market entry`         | **20**          | Medium-High   |
| **hiring**            | `hiring`, `recruiting`, `job openings`, `career opportunities` | **18**          | Medium        |
| **partnership**       | `partnership`, `collaboration`, `joint venture`, `deal`        | **18**          | Medium        |
| **product_launch**    | `launch`, `released`, `new product`, `unveils`, `feature`      | **15**          | Low-Medium    |
| **award**             | `award`, `recognition`, `ranked`, `best`, `top`, `winner`      | **10**          | Low           |

---

## 3. Prospect Score Calculation Formulas

The final Prospect Score ($\mathbf{S_{Prospect}}$) is a composite value normalized to a **0-100** range, used to prioritize outreach.

### A. Total Raw Score ($\mathbf{S_{Total}}$)

The total raw score is the sum of the weights of all unique signals detected plus a sentiment bonus derived from financial news analysis.

$$\mathbf{S_{Total}} = \mathbf{S_{Signal}} + \mathbf{S_{Sentiment}}$$

#### 1. Signal Score ($\mathbf{S_{Signal}}$)

The cumulative sum of the assigned weights for every signal instance found in the news.

$$\mathbf{S_{Signal}} = \sum (\text{Weight of each detected signal})$$

#### 2. Sentiment Bonus ($\mathbf{S_{Sentiment}}$)

A bonus derived from the average AlphaVantage sentiment score (a value between -1 and 1). This ensures only positive sentiment contributes to the score.

$$\mathbf{S_{Sentiment}} = \max \left( 0, \overline{\text{Sentiment Score}} \times 10 \right)$$

- _Note: The bonus is capped at 10 points for perfect positive sentiment and is 0 for neutral or negative sentiment._

### B. Final Normalized Score ($\mathbf{S_{Prospect}}$)

The total raw score is divided by a normalization factor and capped at 100 to fit the standard prioritization scale.

$$\mathbf{S_{Prospect}} = \min \left( 100, \frac{\mathbf{S_{Total}}}{2} \right)$$

- _Note: The division by 2 serves as the primary normalization factor._

---

## 4. Priority and Recommendation Logic

The final score directly maps to an actionable priority level and recommendation for the sales team.

| $\mathbf{S_{Prospect}}$ Range | Priority Level                  | Recommended Action                                                             |
| :---------------------------- | :------------------------------ | :----------------------------------------------------------------------------- |
| $\mathbf{\geq 70}$            | **High Priority - Hot Lead**    | Immediate outreach recommended. Strong signals detected; high buying intent.   |
| $\mathbf{\geq 50}$            | **Medium Priority - Warm Lead** | Schedule outreach within 48 hours. Notable signals detected; good opportunity. |
| $\mathbf{\geq 30}$            | **Low Priority - Cold Lead**    | Add to nurture campaign. Some activity detected; monitor for changes.          |
| $\mathbf{< 30}$               | **Very Low Priority**           | Minimal buying signals. Re-evaluate in 3-6 months.                             |
