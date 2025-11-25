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

# Load environment variables (MONGO_URI, APOLLO_KEY, SERPER_KEY, etc.)
load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGODB_DB_NAME", "ProspectDB")
APOLLO_KEY = os.getenv("APOLLO_API_KEY", "")
SERPER_KEY = os.getenv("SERPER_API_KEY", "")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "demo")
HUNTER_KEY = os.getenv("HUNTER_API_KEY", "")

app = FastAPI(title="Advanced Prospecting Agent")


# Initialize CORS Middleware
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

# Initialize MongoDB Client
try:
    db_client = MongoClient(MONGO_URI)
    db = db_client[MONGO_DB_NAME]
    prospects = db["prospects"]
    # Check connection and create index
    db_client.admin.command('ping')
    prospects.create_index("domain")
    print("✅ MongoDB connected successfully and index created.")
except Exception as e:
    print(f"🚨 MongoDB connection error: {e}")
    # Handle the case where the DB is unavailable gracefully for initial startup


# --- CORE AGENT CLASSES ---

# class DataFetcher:
#     """Fetches data from external APIs (Apollo for firmographics/contacts, Serper/AV for news)"""

#     def get_apollo_data(self, domain: str) -> Dict[str, Any]:
#         """Get company info and prioritized contacts from Apollo"""

#         # NOTE: If APOLLO_KEY is missing or invalid, all Apollo requests will fail with 422 errors.
#         if not APOLLO_KEY:
#              print("🚨 APOLLO_API_KEY is missing. Skipping Apollo requests.")
#              return {
#                  "company_name": domain.split('.')[0].capitalize(),
#                  "industry": "Unknown",
#                  "employee_count": 0,
#                  "contacts": []
#              }

#         # --- Define Standard Headers for Apollo (Using X-Api-Key with correct capitalization) ---
#         headers = {
#             "Content-Type": "application/json",
#             "Cache-Control": "no-cache",
#             "X-Api-Key": APOLLO_KEY
#         }

#         company_name = domain.split('.')[0].capitalize()
#         industry = "Unknown"
#         employee_count = 0

#         # --- 1. Company Enrichment (POST /organizations/enrich) ---
#         company_url = "https://api.apollo.io/v1/organizations/enrich"
#         # Payload now ONLY contains required query parameters (domain)
#         company_payload = {"domain": domain}

#         try:
#             response = requests.post(company_url, json=company_payload, headers=headers, timeout=10)
#             response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
#             company_data = response.json()
#             org = company_data.get("organization", {})

#             company_name = org.get("name", company_name)
#             industry = org.get("industry", "Unknown")
#             employee_count = org.get("estimated_num_employees", 0)

#         except requests.exceptions.RequestException as e:
#             print(f"Apollo company enrichment error: {e}")
#         except Exception as e:
#             print(f"Apollo company enrichment failed: {e}")

#         # --- 2. Get Prioritized Contacts (POST /mixed_people/search) ---
#         contacts_url = "https://api.apollo.io/api/v1/mixed_people/search"

#         # Prioritize key titles for quality outreach
#         target_titles = [
#             "Head of Learning", "VP of Training", "Chief Learning Officer", "Director of Learning",
#             "CEO", "CTO", "VP of Sales", "VP of Engineering", "Head of Strategy", "Director of Technology"
#         ]

#         # Initial search payload with prioritized titles
#         contacts_payload = {
#             "q_organization_domains": domain,
#             "page": 1,
#             "per_page": 15,
#             "person_titles": target_titles
#         }

#         print("contacts_payload", contacts_payload)
#         print("headers", headers)
#         print("contacts_url", contacts_url)

#         contacts = []
#         try:
#             response = requests.post(contacts_url, json=contacts_payload, headers=headers, timeout=10)
#             response.raise_for_status()
#             contacts_data = response.json()

#             for person in contacts_data.get("people", [])[:10]:
#                 contacts.append({
#                     "name": person.get("name", "Unknown"),
#                     "title": person.get("title", ""),
#                     "email": person.get("email", ""),
#                     "linkedin": person.get("linkedin_url", ""),
#                     "apollo_id": person.get("id"),
#                 })
#         except requests.exceptions.RequestException as e:
#             print(f"Apollo contacts search error: {e}")
#         except Exception as e:
#             print(f"Apollo contacts search failed: {e}")

#         # Fallback to a broader search if the targeted search failed or returned few contacts
#         if len(contacts) < 3:
#              print("Falling back to broader contact search...")

#              # Broader titles for general executives/managers
#              contacts_payload["person_titles"] = ["Director", "Manager", "Head", "Analyst"]
#              try:
#                  response = requests.post(contacts_url, json=contacts_payload, headers=headers, timeout=10)
#                  response.raise_for_status()
#                  contacts_data = response.json()
#                  for person in contacts_data.get("people", [])[:10]:
#                      # Only append new contacts, avoiding duplicates
#                      if not any(c.get('apollo_id') == person.get("id") for c in contacts):
#                          contacts.append({
#                              "name": person.get("name", "Unknown"),
#                              "title": person.get("title", ""),
#                              "email": person.get("email", ""),
#                              "linkedin": person.get("linkedin_url", ""),
#                              "apollo_id": person.get("id"),
#                          })
#              except Exception as e:
#                  print(f"Apollo broader contacts search failed: {e}")


#         return {
#             "company_name": company_name,
#             "industry": industry,
#             "employee_count": employee_count,
#             "contacts": contacts
#         }

#     def get_serper_news(self, company_name: str, num_results: int = 20) -> List[Dict[str, Any]]:
#         """Get general and deal-related news articles from Serper (Google News)"""

#         url = "https://google.serper.dev/news"
#         headers = {
#             "X-API-KEY": SERPER_KEY,
#             "Content-Type": "application/json"
#         }

#         # Enhanced query to catch funding/acquisition news
#         query = f'"{company_name}" funding OR "{company_name}" acquisition OR "{company_name}" expansion OR "{company_name}" growth'

#         payload = {
#             "q": query,
#             "num": num_results,
#             "tbs": "qdr:y"
#         }

#         articles = []
#         try:
#             response = requests.post(url, json=payload, headers=headers, timeout=10)
#             response.raise_for_status()
#             data = response.json()

#             for item in data.get("news", []):
#                 articles.append({
#                     "title": item.get("title", ""),
#                     "snippet": item.get("snippet", ""),
#                     "source": item.get("source", ""),
#                     "date": item.get("date", ""),
#                     "link": item.get("link", "")
#                 })
#         except Exception as e:
#             print(f"Serper error: {e}")

#         return articles

#     def get_alphavantage_news(self, ticker: str) -> List[Dict[str, Any]]:
#         """Get financial news and sentiment from AlphaVantage"""

#         url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHAVANTAGE_KEY}&limit=50"

#         articles = []
#         try:
#             response = requests.get(url, timeout=10)
#             response.raise_for_status()
#             data = response.json()

#             for item in data.get('feed', []):
#                 sentiment_data = next((s for s in item.get("ticker_sentiment", []) if s.get("ticker") == ticker), {})

#                 articles.append({
#                     "title": item.get("title", ""),
#                     "url": item.get("url", ""),
#                     "source": item.get("source", ""),
#                     "time_published": item.get("time_published", ""),
#                     "summary": item.get("summary", ""),
#                     "sentiment_score": sentiment_data.get("ticker_sentiment_score", "0"),
#                     "sentiment_label": sentiment_data.get("ticker_sentiment_label", "Neutral")
#                 })
#         except Exception as e:
#             print(f"AlphaVantage error: {e}")

#         return articles

class DataFetcher:
    """Fetches data from external APIs (Hunter/Serper/AV for prospecting data)"""

    def _get_hunter_company_data(self, domain: str) -> Dict[str, Any]:
        """Get company firmographics from Hunter.io Company API"""
        url = "https://api.hunter.io/v2/domain-search"
        params = {
            "domain": domain,
            "api_key": HUNTER_KEY,
            "limit": 10 # Request up to 10 contacts from the domain
        }

        # Default fallback data
        company_data = {
            "company_name": domain.split('.')[0].capitalize(),
            "industry": "Unknown",
            "employee_count": 0,
            "contacts": []
        }

        if not HUNTER_KEY:
             print("🚨 HUNTER_API_KEY is missing. Skipping Hunter requests.")
             return company_data

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", {})

            # Update firmographics
            company_data["company_name"] = data.get("company", company_data["company_name"])

            # Hunter doesn't always provide detailed firmographics on the domain search
            # We'll rely on the contact data it provides

            # Process contacts
            for person in data.get("emails", [])[:10]:
                company_data["contacts"].append({
                    "name": person.get("first_name", "") + " " + person.get("last_name", ""),
                    "title": person.get("position", ""),
                    "email": person.get("value", ""),
                    # Hunter doesn't natively return LinkedIn, but we can search by name/domain later if needed
                    "linkedin": "",
                    "apollo_id": None,
                })

        except requests.exceptions.RequestException as e:
            print(f"Hunter.io company/contact search error: {e}")
        except Exception as e:
            print(f"Hunter.io search failed: {e}")

        return company_data


    # --- New Unified Method to Replace get_apollo_data ---
    def get_prospect_data(self, domain: str) -> Dict[str, Any]:
        """
        Unified function to replace Apollo.io using Hunter.io for company and contact data.
        This function maintains the required output structure.
        """
        return self._get_hunter_company_data(domain) # Hunter's domain search is the best fit for the free tier


    # --- Remaining methods stay the same, but we update the calls later ---
    def get_serper_news(self, company_name: str, num_results: int = 20) -> List[Dict[str, Any]]:
        # ... (Same code as before)
        pass # Placeholder for brevity - the actual method remains unchanged

    def get_alphavantage_news(self, ticker: str) -> List[Dict[str, Any]]:
        # ... (Same code as before)
        pass # Placeholder for brevity - the actual method remains unchanged

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
                         'career opportunities', 'join our team', 'massive hiring']
        },
        'leadership_change': {
            'weight': 22,
            'keywords': ['ceo', 'cto', 'cfo', 'vp', 'appointed', 'hired', 'joins',
                         'new chief', 'executive', 'promotion', 'leadership', 'head of learning', 'chief learning officer']
        },
        'product_launch': {
            'weight': 15,
            'keywords': ['launch', 'released', 'introducing', 'new product', 'unveils',
                         'announces', 'feature', 'innovation', 'beta']
        },
        'partnership': {
            'weight': 18,
            'keywords': ['partnership', 'collaboration', 'joint venture', 'alliance',
                         'agreement', 'deal', 'signed', 'strategic']
        },
        'acquisition': {
            'weight': 25,
            'keywords': ['acquisition', 'acquires', 'merger', 'bought', 'purchased',
                         'takeover', 'm&a', 'integrating']
        },
        'revenue_growth': {
            'weight': 20,
            'keywords': ['revenue', 'profit', 'earnings', 'quarterly results', 'growth',
                         'sales increase', 'record', 'exceeded']
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
            # Concatenate all relevant text fields for comprehensive keyword matching
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
            sentiment_bonus = max(0, avg_sentiment * 10)
            total_score += sentiment_bonus

        # Normalize score to 0-100
        # Scaling factor based on max possible score (e.g., if max possible weight is ~200, use /2)
        # Using a fixed divisor of 2 as a baseline for a moderately complex signal count
        final_score = min(100, total_score / 2)

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
        top_signals_str = ", ".join([f"{s[0].replace('_', ' ')} ({s[1]})" for s in top_signals if s[1] > 0])

        if score >= 70:
            return f"Immediate outreach recommended. Strong signals detected: {top_signals_str}. High buying intent."
        elif score >= 50:
            return f"Schedule outreach within 48 hours. Notable signals: {top_signals_str}. Good opportunity."
        elif score >= 30:
            return f"Add to nurture campaign. Some activity detected: {top_signals_str}. Monitor for changes."
        else:
            return "Low priority. Minimal buying signals. Consider re-evaluating in 3-6 months."

class PersonalizationEngine:
    """Creates personalized content based on prospect data"""

    # NOTE: Customize these templates with your specific pain points and value props!
    PERSONALIZATION_TEMPLATES = {
        'funding_round': """
        Congratulations on your recent **{funding_amount} {funding_round}**!
        With {company} scaling rapidly, I imagine **{pain_point}** is becoming
        an immediate priority for your team.
        """,

        'leadership_change': """
        I saw that {name} recently joined {company} as **{title}**.
        In my experience, new {title_category} leaders often look to
        **{common_initiative}** in their first 90 days, which is precisely where we help.
        """,

        'expansion': """
        Your recent **{signal_detail}** news signals a huge push for growth at {company}.
        That kind of rapid scale often creates challenges in **{pain_point}** which
        we are specialized in solving.
        """
    }

    def personalize_message(self, prospect_data: Dict[str, Any], product_context: str) -> str:
        """
        Creates a personalized message based on the prospect's top signal.
        """

        company_name = prospect_data.get("company_name", "the company")

        # --- SAFE CONTACT & SIGNAL RETRIEVAL ---

        # Safely get the primary contact
        contacts = prospect_data.get("contacts", [])
        # FIX: Check if contacts list is not empty before accessing index [0]
        contact = contacts[0] if contacts else {}
        contact_name = contact.get("name", "Team")

        # Safely get the top signal
        top_signal_list = prospect_data.get("signals")
        # Ensure 'signals' is a list and has at least one element before accessing [0]
        top_signal = top_signal_list[0] if top_signal_list and isinstance(top_signal_list, list) and len(top_signal_list) > 0 else {}
        signal_type = top_signal.get('type')


        # --- 1. Funding Signal Logic ---
        if signal_type == 'funding':
            return self.PERSONALIZATION_TEMPLATES['funding_round'].format(
                funding_amount="significant investment",
                funding_round=top_signal.get('title', 'funding round'),
                company=company_name,
                pain_point="accelerating your engineering capacity or optimizing lead flow" # Example pain point
            ).strip()

        # --- 2. Leadership Change Logic ---
        if signal_type == 'leadership_change' and contact:
            # Simple title category logic
            title = contact.get("title", "Executive")
            title_category = "VP-level" if "VP" in title or "Head" in title else "C-Suite"

            return self.PERSONALIZATION_TEMPLATES['leadership_change'].format(
                name=contact.get("name", "The new leader"),
                company=company_name,
                title=title,
                title_category=title_category,
                common_initiative="implement a clear ROI-focused strategy for the sales stack"
            ).strip()

        # --- 3. Expansion Signal Logic ---
        if signal_type == 'expansion' or signal_type == 'hiring':
            return self.PERSONALIZATION_TEMPLATES['expansion'].format(
                signal_detail=top_signal.get('title', 'growth'),
                company=company_name,
                pain_point="maintaining data quality and integrating new systems efficiently"
            ).strip()

        # --- 4. Default/No Strong Signal ---
        return f"Hi {contact_name},\n\nI noticed {company_name} is in the {prospect_data.get('industry', 'tech')} space and focuses on growth. We specifically help companies like yours **{product_context}**."

class OutreachComposer:
    """Composes multi-touch email sequences"""

    def __init__(self, personalization_engine: PersonalizationEngine):
        self.personalization_engine = personalization_engine

    def _get_primary_contact_name(self, prospect: Dict[str, Any]) -> str:
        """Utility function to safely get the primary contact's name."""
        contacts = prospect.get("contacts", [])
        # FIX: Safely access contact name
        return contacts[0].get("name", "Team") if contacts else "Team"

    def generate_subject_line(self, prospect: Dict[str, Any]) -> str:
        """Generate a personalized subject line based on the score/priority/top signal."""
        company_name = prospect.get("company_name", "Your Company")
        priority = prospect.get("priority", "Low")

        # Safely get top signal type (defaulting to a generic term)
        signals = prospect.get("signals")
        top_signal_type = signals[0].get("type", "Growth") if signals and len(signals) > 0 else "Growth"

        if "High Priority" in priority:
            # Use top signal for high urgency
            if top_signal_type == 'funding':
                return f"Congrats on {company_name}'s funding + quick question"
            return f"Idea for {company_name}'s {top_signal_type.replace('_', ' ')} strategy"

        if "Medium Priority" in priority:
            return f"Idea for {company_name}'s {prospect.get('industry', 'industry')} strategy"

        return f"Resource for {company_name}'s growth"

    def create_initial_outreach(self, prospect: Dict[str, Any]):

        # Define context for your product (CUSTOMIZE THIS!)
        # Example for e-learning target: "building highly engaging, mobile-first learning modules."
        product_context = "streamline their lead research and scoring process to close deals faster."

        # Get the personalized opening
        personalized_opening = self.personalization_engine.personalize_message(prospect, product_context)

        # Get the top contact name (using the safe utility function)
        contact_name = self._get_primary_contact_name(prospect)

        # Add a clear line break between the personalized line and the pitch
        body = (
            f"Hi {contact_name},\n\n"
            f"{personalized_opening}\n\n"
            f"My team specializes in {product_context}, and based on your current high priority signals, "
            "I believe a quick 15-minute chat could clarify where we can specifically assist your team.\n\n"
            "Best,\n[Your Name]"
        )

        email = {
            'subject': self.generate_subject_line(prospect),
            'body': body,
            'call_to_action': "Are you open to a quick 15-minute conversation on Thursday or Friday?"
        }
        return email

    def create_follow_up(self, prospect: Dict[str, Any], days_after: int):
        """Creates a simple, bump-style follow-up."""

        # Safely get contact name
        contact_name = self._get_primary_contact_name(prospect)
        subject = f"Re: {self.generate_subject_line(prospect)}"

        # Reference the original personalized signal
        signals = prospect.get("signals")
        original_signal_type = signals[0].get("type", "growth") if signals and len(signals) > 0 else "growth"

        body = (
            f"Hi {contact_name},\n\n"
            f"Just bumping this up. Did you have a chance to look at the quick note I sent regarding {prospect.get('company_name')}'s {original_signal_type.replace('_', ' ')}?\n\n"
            f"Let me know your availability.\n\n"
            "Best,\n[Your Name]"
        )
        return {'subject': subject, 'body': body, 'call_to_action': "Let me know your availability."}

    def create_value_add(self, prospect: Dict[str, Any], days_after: int):
        """Creates a value-add email using a general signal."""
        signal_summary = prospect.get("signal_summary", {})

        # Focus on the most prevalent pain point/opportunity
        if signal_summary.get('funding', 0) > 0 or signal_summary.get('expansion', 0) > 0:
             focus = "accelerating post-funding growth"
        elif signal_summary.get('leadership_change', 0) > 0:
             focus = "onboarding new leadership initiatives"
        else:
             focus = "efficiency"

        # Safely get contact name
        contact_name = self._get_primary_contact_name(prospect)
        subject = f"A quick resource for {prospect.get('company_name')} focusing on {focus}"
        body = (
            f"Hi {contact_name},\n\n"
            f"Given {prospect.get('company_name')}'s current focus on {focus}, I thought this recent **Case Study** "
            "on how similar companies achieved [X result] would be helpful. [Insert link to blog post/case study here].\n\n"
            "If it sparks any questions, feel free to reply.\n\nBest,\n[Your Name]"
        )
        return {'subject': subject, 'body': body, 'call_to_action': "N/A"}

    def create_breakup_email(self, prospect: Dict[str, Any], days_after: int):
        """Creates a final 'break-up' email."""

        # Safely get contact name
        contact_name = self._get_primary_contact_name(prospect)
        company_name = prospect.get("company_name", "Your Company")

        subject = f"Closing the loop with {company_name}"
        body = (
            f"Hi {contact_name},\n\n"
            "I know you're busy, so this will be my final outreach. I truly believe our solution could have saved your team "
            "time and resources on the exact challenge I mentioned in my first email.\n\n"
            "I'll take this as a 'not right now,' but feel free to reach out if priorities shift. Wishing you success,\n\n"
            "[Your Name]"
        )
        return {'subject': subject, 'body': body, 'call_to_action': "N/A"}

    def compose_sequence(self, prospect: Dict[str, Any], campaign_type='standard'):
        """Composes multi-touch email sequences"""
        sequence = {
            'email_1_initial_outreach': self.create_initial_outreach(prospect),
            'email_2_follow_up': self.create_follow_up(prospect, days_after=3),
            'email_3_value_add': self.create_value_add(prospect, days_after=7),
            'email_4_breakup': self.create_breakup_email(prospect, days_after=14)
        }
        return sequence

# Initialize all engines
fetcher = DataFetcher()
signal_engine = SignalEngine()
personalization_engine = PersonalizationEngine()
# Outreach Composer needs the personalization engine
outreach_composer = OutreachComposer(personalization_engine)


# --- API ROUTES ---

# @app.post("/research")
# def research_company(domain: str, ticker: str = None):
#     """
#     Step 1: Research a company by domain. Fetches company info and contacts
#     from Apollo, and news/signals from Serper/AlphaVantage.
#     """

#     if not domain:
#         raise HTTPException(status_code=400, detail="Domain is required")

#     # Generate unique ID
#     research_id = str(uuid.uuid4())

#     print(f"🔍 Starting research for {domain}...")

#     # Fetch Apollo data
#     apollo_data = fetcher.get_apollo_data(domain)
#     company_name = apollo_data["company_name"]

#     # Fetch news from multiple sources
#     serper_articles = fetcher.get_serper_news(company_name, num_results=20)

#     alphavantage_articles = []
#     if ticker:
#         alphavantage_articles = fetcher.get_alphavantage_news(ticker)

#     # Combine all articles
#     all_articles = serper_articles + alphavantage_articles

#     # Save to database
#     record = {
#         "_id": research_id,
#         "domain": domain,
#         "ticker": ticker,
#         "company_name": company_name,
#         "industry": apollo_data["industry"],
#         "employee_count": apollo_data["employee_count"],
#         "contacts": apollo_data["contacts"],
#         "articles": all_articles,
#         "researched_at": datetime.now(),
#         "total_articles": len(all_articles)
#     }

#     try:
#         prospects.insert_one(record)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Database insert failed: {e}")

#     print(f"✅ Research completed for {domain}")

#     return {
#         "research_id": research_id,
#         "domain": domain,
#         "company_name": company_name,
#         "industry": apollo_data["industry"],
#         "total_contacts": len(apollo_data["contacts"]),
#         "contacts": apollo_data["contacts"],
#         "total_articles": len(all_articles),
#         "message": f"Research complete. Found {len(all_articles)} articles and {len(apollo_data['contacts'])} contacts."
#     }

@app.post("/research")
def research_company(domain: str, ticker: str = None):
    """
    Step 1: Research a company by domain. Fetches company info and contacts
    from Hunter.io, and news/signals from Serper/AlphaVantage.
    """

    if not domain:
        raise HTTPException(status_code=400, detail="Domain is required")

    # ... (rest of setup)
    # Generate unique ID
    research_id = str(uuid.uuid4())

    print(f"🔍 Starting research for {domain}...")

    # Fetch data using the NEW unified function
    # CHANGE: apollo_data -> prospect_data
    prospect_data = fetcher.get_prospect_data(domain)
    company_name = prospect_data["company_name"]

    # Fetch news from multiple sources
    serper_articles = fetcher.get_serper_news(company_name, num_results=20)

    alphavantage_articles = []
    if ticker:
        alphavantage_articles = fetcher.get_alphavantage_news(ticker)

#     # Combine all articles
    all_articles = serper_articles + alphavantage_articles

    # Save to database
    record = {
        "_id": research_id,
        "domain": domain,
        "ticker": ticker,
        "company_name": company_name,
        "industry": prospect_data["industry"], # Changed to prospect_data
        "employee_count": prospect_data["employee_count"], # Changed to prospect_data
        "contacts": prospect_data["contacts"], # Changed to prospect_data
        "articles": all_articles,
        "researched_at": datetime.now(),
        "total_articles": len(all_articles)
    }

    # ... (rest of function)

    return {
        "research_id": research_id,
        "domain": domain,
        "company_name": company_name,
        "industry": prospect_data["industry"],
        "total_contacts": len(prospect_data["contacts"]),
        "contacts": prospect_data["contacts"],
        "total_articles": len(all_articles),
        "message": f"Research complete. Found {len(all_articles)} articles and {len(prospect_data['contacts'])} contacts."
    }

@app.get("/signals/{research_id}")
def get_signals(research_id: str):
    """
    Step 2: Get signals and scoring for a researched company.
    Analyzes articles and assigns a priority score.
    """

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
        "prospect_score": signal_data["prospect_score"],
        "priority": signal_data["priority"],
        "recommendation": signal_data["recommendation"],
        "total_signals": signal_data["total_signals"],
        "signal_summary": signal_data["signal_summary"],
        "top_signals": signal_data["signals"][:5], # Return only top 5 signals
        "articles_analyzed": len(articles)
    }

@app.get("/compose/{research_id}")
def compose_outreach(research_id: str):
    """
    Step 3: Generates a full, personalized outreach sequence based on the
    finalized prospect score and signals.
    """

    # 1. Fetch the FULL, analyzed record from the database
    record = prospects.find_one({"_id": research_id})

    if not record:
        raise HTTPException(status_code=404, detail="Research ID not found.")

    # 2. Check if the record has been scored (signals generated)
    if "prospect_score" not in record:
        # Run signals analysis automatically if not already done
        articles = record.get("articles", [])
        signal_data = signal_engine.detect_signals(articles)
        record.update(signal_data) # Update the local record for composing

        # Update the database in the background
        prospects.update_one(
            {"_id": research_id},
            {"$set": {
                "signals": signal_data["signals"],
                "prospect_score": signal_data["prospect_score"],
                "priority": signal_data["priority"],
                "analyzed_at": datetime.now()
            }}
        )
        print("💡 Signals generated automatically for outreach composition.")

    # 3. Compose the full sequence
    outreach_sequence = outreach_composer.compose_sequence(record)

    # 4. Return the composed sequence

    # FIX: Safely retrieve the primary contact, defaulting to an empty dict if the list is empty
    contacts = record.get("contacts", [])
    primary_contact = contacts[0] if contacts else {}

    return {
        "research_id": research_id,
        "company_name": record.get("company_name"),
        "prospect_score": record.get("prospect_score"),
        "priority": record.get("priority"),
        "target_contact": primary_contact, # <--- FIXED LINE
        "sequence": outreach_sequence
    }

# --- Utility Routes (For Frontend/Management) ---

@app.get("/company/{domain}")
def get_company_details(domain: str):
    """Get full company details including contacts and articles by domain."""

    record = prospects.find_one({"domain": domain}, sort=[("researched_at", -1)])

    if not record:
        raise HTTPException(status_code=404, detail=f"No research found for domain: {domain}")

    # Ensure _id is a string before returning
    record["research_id"] = str(record.pop("_id"))

    return record

@app.get("/companies")
def get_all_companies(limit: int = 20):
    """
    Get a list of the most recently researched companies.
    """

    cursor = prospects.find({}).sort("researched_at", -1).limit(limit)

    company_list = list(cursor)

    if not company_list:
        return [] # Return empty list instead of 404 if no data

    for company in company_list:
        # Rename _id to research_id
        company["research_id"] = str(company.pop("_id"))

    return company_list


# Run with: uvicorn app:app --reload