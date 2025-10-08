import os
import json
import uuid
import time
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from typing import Dict, List, Any
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGODB_DB_NAME", "ProspectDB")
APOLLO_KEY = os.getenv("APOLLO_API_KEY", "")
SERPER_KEY = os.getenv("SERPER_API_KEY", "")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "demo")


app = FastAPI(title="Advanced Prospecting Agent")


# Initialize
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_client = MongoClient(MONGO_URI)
db = db_client[MONGO_DB_NAME]
prospects = db["prospects"]

# Create index for faster queries
prospects.create_index("domain")

class DataFetcher:
    """Fetches data from external APIs"""

    def get_apollo_data(self, domain: str) -> Dict[str, Any]:
        """Get company info and contacts from Apollo"""

        # Company enrichment
        company_url = "https://api.apollo.io/v1/organizations/enrich"
        headers = {"Content-Type": "application/json", "Cache-Control": "no-cache"}

        company_payload = {"domain": domain, "api_key": APOLLO_KEY}

        try:
            response = requests.post(company_url, json=company_payload, headers=headers, timeout=10)
            company_data = response.json()

            company_name = company_data.get("organization", {}).get("name", domain.split('.')[0].capitalize())
            industry = company_data.get("organization", {}).get("industry", "Unknown")
            employee_count = company_data.get("organization", {}).get("estimated_num_employees", 0)

        except Exception as e:
            print(f"Apollo company error: {e}")
            company_name = domain.split('.')[0].capitalize()
            industry = "Unknown"
            employee_count = 0

        # Get contacts
        contacts_url = "https://api.apollo.io/v1/mixed_people/search"
        contacts_payload = {
            "api_key": APOLLO_KEY,
            "q_organization_domains": domain,
            "page": 1,
            "per_page": 10,
            "person_titles": ["CEO", "CTO", "VP", "Director", "Manager", "Head"]
        }

        contacts = []
        try:
            response = requests.post(contacts_url, json=contacts_payload, headers=headers, timeout=10)
            contacts_data = response.json()

            for person in contacts_data.get("people", [])[:10]:
                contacts.append({
                    "name": person.get("name", "Unknown"),
                    "title": person.get("title", ""),
                    "email": person.get("email", ""),
                    "linkedin": person.get("linkedin_url", "")
                })
        except Exception as e:
            print(f"Apollo contacts error: {e}")

        return {
            "company_name": company_name,
            "industry": industry,
            "employee_count": employee_count,
            "contacts": contacts
        }

    def get_serper_news(self, company_name: str, num_results: int = 20) -> List[Dict[str, Any]]:
        """Get news articles from Serper (Google News)"""

        url = "https://google.serper.dev/news"
        headers = {
            "X-API-KEY": SERPER_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "q": f'"{company_name}" OR "{company_name.split()[0]}"',
            "num": num_results,
            "tbs": "qdr:y"  # Last year
        }

        articles = []
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            data = response.json()

            for item in data.get("news", []):
                articles.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("source", ""),
                    "date": item.get("date", ""),
                    "link": item.get("link", "")
                })
        except Exception as e:
            print(f"Serper error: {e}")

        return articles

    def get_alphavantage_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Get news and sentiment from AlphaVantage"""

        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHAVANTAGE_KEY}&limit=50"

        articles = []
        try:
            response = requests.get(url, timeout=10)
            data = response.json()

            for item in data.get('feed', []):
                sentiment_data = next((s for s in item.get("ticker_sentiment", []) if s.get("ticker") == ticker), {})

                articles.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", ""),
                    "time_published": item.get("time_published", ""),
                    "summary": item.get("summary", ""),
                    "sentiment_score": sentiment_data.get("ticker_sentiment_score", "0"),
                    "sentiment_label": sentiment_data.get("ticker_sentiment_label", "Neutral")
                })
        except Exception as e:
            print(f"AlphaVantage error: {e}")

        return articles

class SignalEngine:
    """Advanced signal detection and scoring engine"""

    # Define buying signals with weights
    SIGNALS = {
        'funding': {
            'weight': 25,
            'keywords': ['funding', 'raised', 'investment', 'series a', 'series b', 'series c',
                        'venture capital', 'vc funding', 'seed round', 'investors', 'valuation']
        },
        'expansion': {
            'weight': 20,
            'keywords': ['expansion', 'expanding', 'new office', 'opening', 'growing',
                        'international', 'market entry', 'launch']
        },
        'hiring': {
            'weight': 18,
            'keywords': ['hiring', 'recruiting', 'talent', 'job openings', 'positions',
                        'career opportunities', 'join our team']
        },
        'leadership_change': {
            'weight': 22,
            'keywords': ['ceo', 'cto', 'cfo', 'vp', 'appointed', 'hired', 'joins',
                        'new chief', 'executive', 'promotion', 'leadership']
        },
        'product_launch': {
            'weight': 15,
            'keywords': ['launch', 'released', 'introducing', 'new product', 'unveils',
                        'announces', 'feature', 'innovation']
        },
        'partnership': {
            'weight': 18,
            'keywords': ['partnership', 'collaboration', 'joint venture', 'alliance',
                        'agreement', 'deal', 'signed']
        },
        'acquisition': {
            'weight': 25,
            'keywords': ['acquisition', 'acquires', 'merger', 'bought', 'purchased',
                        'takeover', 'm&a']
        },
        'revenue_growth': {
            'weight': 20,
            'keywords': ['revenue', 'profit', 'earnings', 'quarterly results', 'growth',
                        'sales increase', 'record']
        },
        'award': {
            'weight': 10,
            'keywords': ['award', 'recognition', 'ranked', 'best', 'top', 'winner', 'excellence']
        }
    }

    def detect_signals(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect buying signals from articles"""

        detected_signals = []
        signal_counts = {signal: 0 for signal in self.SIGNALS.keys()}
        total_score = 0

        for article in articles:
            text = (article.get("title", "") + " " +
                   article.get("snippet", "") + " " +
                   article.get("summary", "")).lower()

            article_signals = []

            for signal_type, config in self.SIGNALS.items():
                if any(keyword in text for keyword in config['keywords']):
                    if signal_type not in article_signals:
                        article_signals.append(signal_type)
                        signal_counts[signal_type] += 1

                        detected_signals.append({
                            'type': signal_type,
                            'title': article.get("title", ""),
                            'source': article.get("source", ""),
                            'date': article.get("date", "") or article.get("time_published", ""),
                            'url': article.get("link", "") or article.get("url", ""),
                            'weight': config['weight']
                        })

        # Calculate composite score
        for signal in detected_signals:
            total_score += signal['weight']

        # Add sentiment bonus from AlphaVantage
        sentiment_scores = [float(a.get("sentiment_score", 0)) for a in articles
                           if a.get("sentiment_score") and a.get("sentiment_score") != "0"]

        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            sentiment_bonus = max(0, avg_sentiment * 10)  # Positive sentiment adds up to 10 points
            total_score += sentiment_bonus

        # Normalize score to 0-100
        final_score = min(100, total_score / 2)  # Divide by 2 for normalization

        # Determine priority
        if final_score >= 70:
            priority = "High Priority - Hot Lead"
        elif final_score >= 50:
            priority = "Medium Priority - Warm Lead"
        elif final_score >= 30:
            priority = "Low Priority - Cold Lead"
        else:
            priority = "Very Low Priority - Research Only"

        return {
            "signals": detected_signals,
            "signal_summary": signal_counts,
            "total_signals": len(detected_signals),
            "prospect_score": round(final_score, 2),
            "priority": priority,
            "recommendation": self._get_recommendation(signal_counts, final_score)
        }

    def _get_recommendation(self, signal_counts: Dict, score: float) -> str:
        """Generate actionable recommendation"""

        top_signals = sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_signals_str = ", ".join([f"{s[0]} ({s[1]})" for s in top_signals if s[1] > 0])

        if score >= 70:
            return f"Immediate outreach recommended. Strong signals detected: {top_signals_str}. High buying intent."
        elif score >= 50:
            return f"Schedule outreach within 48 hours. Notable signals: {top_signals_str}. Good opportunity."
        elif score >= 30:
            return f"Add to nurture campaign. Some activity detected: {top_signals_str}. Monitor for changes."
        else:
            return "Low priority. Minimal buying signals. Consider re-evaluating in 3-6 months."

# Initialize
fetcher = DataFetcher()
signal_engine = SignalEngine()


# API Routes
@app.post("/research")
def research_company(domain: str, ticker: str = None):
    """Research a company by domain"""

    if not domain:
        raise HTTPException(status_code=400, detail="Domain is required")

    # Generate unique ID
    research_id = str(uuid.uuid4())

    print(f"🔍 Starting research for {domain}...")

    # Fetch Apollo data
    apollo_data = fetcher.get_apollo_data(domain)
    company_name = apollo_data["company_name"]

    # Fetch news from multiple sources
    serper_articles = fetcher.get_serper_news(company_name, num_results=20)

    alphavantage_articles = []
    if ticker:
        alphavantage_articles = fetcher.get_alphavantage_news(ticker)

    # Combine all articles
    all_articles = serper_articles + alphavantage_articles

    # Save to database
    record = {
        "_id": research_id,
        "domain": domain,
        "ticker": ticker,
        "company_name": company_name,
        "industry": apollo_data["industry"],
        "employee_count": apollo_data["employee_count"],
        "contacts": apollo_data["contacts"],
        "articles": all_articles,
        "researched_at": datetime.now(),
        "total_articles": len(all_articles)
    }

    prospects.insert_one(record)

    print(f"✅ Research completed for {domain}")

    return {
        "research_id": research_id,
        "domain": domain,
        "company_name": company_name,
        "industry": apollo_data["industry"],
        "employee_count": apollo_data["employee_count"],
        "total_contacts": len(apollo_data["contacts"]),
        "contacts": apollo_data["contacts"],
        "total_articles": len(all_articles),
        "articles": all_articles[:10],  # Return first 10 articles
        "message": f"Research complete. Found {len(all_articles)} articles and {len(apollo_data['contacts'])} contacts."
    }

@app.get("/signals/{research_id}")
def get_signals(research_id: str):
    """Get signals and scoring for a researched company"""

    record = prospects.find_one({"_id": research_id})

    if not record:
        raise HTTPException(status_code=404, detail="Research ID not found")

    # Run signal detection
    articles = record.get("articles", [])
    signal_data = signal_engine.detect_signals(articles)

    # Update record with signals
    prospects.update_one(
        {"_id": research_id},
        {"$set": {
            "signals": signal_data["signals"],
            "signal_summary": signal_data["signal_summary"],
            "prospect_score": signal_data["prospect_score"],
            "priority": signal_data["priority"],
            "recommendation": signal_data["recommendation"],
            "analyzed_at": datetime.now()
        }}
    )

    return {
        "research_id": research_id,
        "company_name": record.get("company_name"),
        "domain": record.get("domain"),
        "prospect_score": signal_data["prospect_score"],
        "priority": signal_data["priority"],
        "recommendation": signal_data["recommendation"],
        "total_signals": signal_data["total_signals"],
        "signal_summary": signal_data["signal_summary"],
        "signals": signal_data["signals"],
        "articles_analyzed": len(articles)
    }

@app.get("/signals/domain/{domain}")
def get_signals_by_domain(domain: str):
    """Get signals by domain (alternative to research_id)"""

    record = prospects.find_one({"domain": domain}, sort=[("researched_at", -1)])

    if not record:
        raise HTTPException(status_code=404, detail=f"No research found for domain: {domain}")

    research_id = record["_id"]

    # Run signal detection if not already done
    if "signals" not in record:
        articles = record.get("articles", [])
        signal_data = signal_engine.detect_signals(articles)

        prospects.update_one(
            {"_id": research_id},
            {"$set": {
                "signals": signal_data["signals"],
                "signal_summary": signal_data["signal_summary"],
                "prospect_score": signal_data["prospect_score"],
                "priority": signal_data["priority"],
                "recommendation": signal_data["recommendation"],
                "analyzed_at": datetime.now()
            }}
        )
    else:
        signal_data = {
            "signals": record.get("signals", []),
            "signal_summary": record.get("signal_summary", {}),
            "prospect_score": record.get("prospect_score", 0),
            "priority": record.get("priority", ""),
            "recommendation": record.get("recommendation", ""),
            "total_signals": len(record.get("signals", []))
        }

    return {
        "research_id": research_id,
        "company_name": record.get("company_name"),
        "domain": domain,
        "prospect_score": signal_data["prospect_score"],
        "priority": signal_data["priority"],
        "recommendation": signal_data["recommendation"],
        "total_signals": signal_data["total_signals"],
        "signal_summary": signal_data["signal_summary"],
        "signals": signal_data["signals"]
    }

@app.get("/company/{domain}")
def get_company_details(domain: str):
    """Get full company details including contacts and articles"""

    record = prospects.find_one({"domain": domain}, sort=[("researched_at", -1)])

    if not record:
        raise HTTPException(status_code=404, detail=f"No research found for domain: {domain}")

    # record.pop("_id")

    return record

@app.get("/companies")
def get_all_companies(limit: int = 20):
    """
    Get a list of the most recently researched companies.

    Returns:
        A list of company records suitable for a history table.
    """

    cursor = prospects.find({}).sort("researched_at", -1).limit(limit)

    company_list = list(cursor)

    if not company_list:
        # Check if the list is empty
        raise HTTPException(status_code=404, detail="No companies found in the database.")

    for company in company_list:
        # Convert ObjectId (which holds the research_id string) to a standard string and rename it.
        company["research_id"] = str(company.pop("_id"))

    return company_list



# Run with: uvicorn app:app --reload