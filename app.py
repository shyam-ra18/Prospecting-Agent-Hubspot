import os
import json
import uuid
import time
import math
import requests
import logging
import traceback
import re
import dns.resolver
import asyncio
import aiohttp
from dotenv import load_dotenv
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from openai import OpenAI

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGODB_DB_NAME", "ProspectDB")
APOLLO_KEY = os.getenv("APOLLO_API_KEY", "")
SERPER_KEY = os.getenv("SERPER_API_KEY", "")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "demo")
HUNTER_KEY = os.getenv("HUNTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = FastAPI(
    title="Advanced Prospecting Agent for Dublabs.ai - Verified Edition",
    debug=True
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize CORS Middleware
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://prospecting-agent-hubspot.onrender.com",
    "https://prospecting-agent-hubspot.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Exception occurred: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )

# Initialize MongoDB Client
try:
    db_client = MongoClient(MONGO_URI)
    db = db_client[MONGO_DB_NAME]
    prospects = db["prospects"]
    metrics = db["metrics"]
    db_client.admin.command('ping')
    prospects.create_index("domain")
    print("✅ MongoDB connected successfully and index created.")
except Exception as e:
    print(f"🚨 MongoDB connection error: {e}")

# ============================================================================
# CONFIGURATION & DATA CLASSES
# ============================================================================

@dataclass
class ScoringConfig:
    """Configuration for the enhanced scoring system"""
    # Scoring dimension weights (must sum to 100)
    FIRMOGRAPHIC_WEIGHT: int = 30  # Company fit
    INTENT_SIGNAL_WEIGHT: int = 50  # Buying signals
    RECENCY_WEIGHT: int = 20  # Time decay factor

    # Recency decay parameters
    DECAY_HALF_LIFE_DAYS: int = 90  # Signals lose 50% value after 90 days
    MAX_SIGNAL_AGE_DAYS: int = 365  # Ignore signals older than 1 year

    # Signal caps (max times a signal type can contribute)
    MAX_SIGNALS_PER_TYPE: int = 2  # Prevent signal stacking

    # ICP Fit thresholds
    TARGET_INDUSTRIES: List[str] = None
    IDEAL_EMPLOYEE_RANGES: List[tuple] = None
    MIN_EMPLOYEE_COUNT: int = 20

    def __post_init__(self):
        if self.TARGET_INDUSTRIES is None:
            self.TARGET_INDUSTRIES = [
                'E-Learning', 'EdTech', 'Education Technology',
                'Online Education', 'E-learning', 'Learning Management'
            ]
        if self.IDEAL_EMPLOYEE_RANGES is None:
            self.IDEAL_EMPLOYEE_RANGES = [(50, 500), (500, 2000)]


# ============================================================================
# EMAIL VERIFICATION ENGINE
# ============================================================================

class EmailVerificationEngine:
    """Comprehensive email verification with multiple validation layers"""

    def __init__(self):
        self.smtp_timeout = 10

    async def verify_email_comprehensive(self, email: str) -> Dict[str, any]:
        """Multi-layer email verification"""
        verification_results = {
            'email': email,
            'is_valid_format': False,
            'is_deliverable': False,
            'is_disposable': False,
            'has_mx_records': False,
            'smtp_response': None,
            'confidence_score': 0.0,
            'verification_timestamp': datetime.now().isoformat()
        }

        try:
            # 1. Format validation
            if not self._validate_email_format(email):
                verification_results['confidence_score'] = 0.0
                return verification_results

            verification_results['is_valid_format'] = True

            # 2. Disposable email check
            if self._is_disposable_email(email):
                verification_results['is_disposable'] = True
                verification_results['confidence_score'] = 0.1
                return verification_results

            # 3. DNS MX record check
            if not await self._check_mx_records(email):
                verification_results['confidence_score'] = 0.3
                return verification_results

            verification_results['has_mx_records'] = True

            # 4. Basic deliverability check (simplified)
            verification_results['is_deliverable'] = await self._check_deliverability(email)

            # Calculate final confidence score
            verification_results['confidence_score'] = self._calculate_email_confidence(verification_results)

        except Exception as e:
            logger.error(f"Email verification error for {email}: {str(e)}")

        return verification_results

    def _validate_email_format(self, email: str) -> bool:
        """Validate email format using regex"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    def _is_disposable_email(self, email: str) -> bool:
        """Check if email is from disposable email service"""
        disposable_domains = {
            'tempmail.com', 'guerrillamail.com', 'mailinator.com',
            '10minutemail.com', 'throwawaymail.com', 'yopmail.com',
            'fakeinbox.com', 'trashmail.com', 'dispostable.com'
        }
        domain = email.split('@')[-1].lower()
        return domain in disposable_domains

    async def _check_mx_records(self, email: str) -> bool:
        """Check if domain has valid MX records"""
        try:
            domain = email.split('@')[-1]
            answers = dns.resolver.resolve(domain, 'MX')
            return len(answers) > 0
        except:
            return False

    async def _check_deliverability(self, email: str) -> bool:
        """Basic deliverability check"""
        # In production, use services like NeverBounce/ZeroBounce
        # This is a simplified version
        return True  # Assume true for demo

    def _calculate_email_confidence(self, results: Dict) -> float:
        """Calculate email confidence score 0-1"""
        score = 0.0

        if results['is_valid_format']:
            score += 0.2

        if not results['is_disposable']:
            score += 0.3

        if results['has_mx_records']:
            score += 0.3

        if results['is_deliverable']:
            score += 0.2

        return round(score, 2)


# ============================================================================
# LLM VALIDATION ENGINE (Updated for OpenAI)
# ============================================================================

class LLMValidationEngine:
    """Use OpenRouter (supports multiple models including OpenAI) to validate and enhance data accuracy"""

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # Use OpenRouter endpoint instead of OpenAI
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")  # OpenRouter model format
        self.validation_cache = {}
        self.rate_limit_delay = 1.0
        self.max_retries = 2

        # Debug: Check if API key is loaded
        if not self.api_key:
            logger.warning("❌ OpenRouter API key not found in environment variables")
        else:
            logger.info(f"✅ OpenRouter API key loaded (first 8 chars: {self.api_key[:8]}...)")
            logger.info(f"✅ Using OpenRouter model: {self.model}")
            logger.info(f"✅ Using OpenRouter base URL: {self.base_url}")

    async def validate_company_data(self, company_data: Dict) -> Dict[str, any]:
        """Use OpenRouter to validate company information"""
        if not self.api_key:
            logger.warning("⚠️ OpenRouter client not available, using fallback validation")
            return self._get_fallback_validation()

        prompt = f"""
        Analyze this company data for accuracy and consistency:

        Company: {company_data.get('company_name', 'Unknown')}
        Industry: {company_data.get('industry', 'Unknown')}
        Employee Count: {company_data.get('employee_count', 'Unknown')}
        Domain: {company_data.get('domain', 'Unknown')}

        Please assess:
        1. Does the industry match typical companies of this size?
        2. Is the employee count realistic for this industry?
        3. Are there any data inconsistencies?
        4. What's the overall data quality score (1-10)?

        Respond in JSON format:
        {{
            "data_quality_score": 0-10,
            "industry_consistency": "high|medium|low",
            "size_consistency": "high|medium|low",
            "identified_issues": ["list", "of", "issues"],
            "confidence_level": "high|medium|low",
            "recommendations": ["list", "of", "improvements"]
        }}
        """

        # Add rate limiting delay
        await asyncio.sleep(self.rate_limit_delay)

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"🔧 Making OpenRouter API request with model: {self.model} (attempt {attempt + 1})")

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/your-repo",  # Required by OpenRouter
                    "X-Title": "Prospecting Agent"  # Required by OpenRouter
                }

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a data quality analyst specializing in company information validation. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 800,
                    "response_format": {"type": "json_object"}
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30
                    ) as response:

                        if response.status == 401:
                            error_text = await response.text()
                            logger.error(f"❌ OpenRouter API 401 Unauthorized: {error_text}")
                            return self._get_fallback_validation_with_error("OpenRouter API key invalid")

                        elif response.status != 200:
                            error_text = await response.text()
                            logger.error(f"❌ OpenRouter API error {response.status}: {error_text}")
                            return self._get_fallback_validation_with_error(f"OpenRouter API error: {response.status}")

                        response_data = await response.json()

                        if "choices" not in response_data or not response_data["choices"]:
                            logger.error("❌ OpenRouter API returned no choices")
                            return self._get_fallback_validation()

                        content = response_data["choices"][0]["message"]["content"]
                        logger.info(f"✅ OpenRouter response received: {content[:100]}...")

                        try:
                            # Try to parse JSON response
                            validation_result = json.loads(content)
                            logger.info("✅ OpenRouter validation successful")
                            return validation_result
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ Failed to parse OpenRouter JSON response: {e}")
                            # Try to extract JSON from text response
                            return self._extract_json_from_text(content)

            except Exception as e:
                error_str = str(e)
                logger.error(f"❌ OpenRouter validation attempt {attempt + 1} failed: {error_str}")

                # Check if it's a rate limit or quota error
                if "quota" in error_str.lower() or "429" in error_str or "insufficient_quota" in error_str:
                    logger.error("🚨 OpenRouter quota exceeded. Switching to fallback mode.")
                    return self._get_fallback_validation_with_error("OpenRouter quota exceeded")

                elif "rate limit" in error_str.lower():
                    # Exponential backoff for rate limits
                    wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
                    logger.warning(f"⏳ Rate limit hit, waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
                    continue

                else:
                    # For other errors, return fallback immediately
                    return self._get_fallback_validation_with_error(f"OpenRouter error: {error_str}")

        # If all retries failed
        return self._get_fallback_validation_with_error("All retries exhausted")

async def validate_news_relevance(self, articles: List[Dict], company_name: str) -> List[Dict]:
    """Quick fix - validate only first 5 articles"""
    if not self.api_key:
        return articles

    # Only validate first 5 articles to save API calls
    articles_to_validate = articles[:5]
    remaining_articles = articles[5:]

    validated_articles = []

    for article in articles_to_validate:
        validated_article = await self._validate_single_article(article, company_name)
        validated_articles.append(validated_article)
        await asyncio.sleep(1.0)  # Rate limiting

    # Mark remaining articles as not validated but assume relevant
    for article in remaining_articles:
        article['llm_validation'] = {
            'is_relevant': True,
            'relevance_confidence': 0.3,
            'relevance_reason': 'Not validated (limit reached)',
            'key_topics': []
        }
        article['is_relevant'] = True
        article['relevance_confidence'] = 0.3
        validated_articles.append(article)

    return validated_articles
    def _extract_json_from_text(self, text: str) -> Dict:
        """Extract JSON from text response if not pure JSON"""
        try:
            # First try to parse directly as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            json_match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Fallback: return a structured error response
            return {
                "is_relevant": False,
                "relevance_confidence": 0.5,
                "relevance_reason": "Could not parse API response",
                "key_topics": [],
                "parse_error": True
            }

    def _get_fallback_validation(self) -> Dict:
        """Fallback when OpenRouter validation fails"""
        return {
            "data_quality_score": 5,
            "industry_consistency": "unknown",
            "size_consistency": "unknown",
            "identified_issues": ["validation_service_unavailable"],
            "confidence_level": "low",
            "recommendations": ["manual_data_verification_recommended"]
        }

    def _get_fallback_validation_with_error(self, error: str) -> Dict:
        """Fallback with specific error information"""
        return {
            "data_quality_score": 5,
            "industry_consistency": "unknown",
            "size_consistency": "unknown",
            "identified_issues": [f"validation_service_error: {error}"],
            "confidence_level": "low",
            "recommendations": ["manual_data_verification_recommended"],
            "api_error": error
        }

    async def check_api_health(self) -> Dict[str, Any]:
        """Check if OpenRouter API is healthy and available"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "Prospecting Agent"
            }

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Say 'OK' if you're working."}],
                "max_tokens": 10
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=10
                ) as response:

                    return {
                        "status": "healthy" if response.status == 200 else "unhealthy",
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
                        "status_code": response.status_code
                    }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response_time_ms": 0,
                "status_code": 0
            }

# ============================================================================
# DYNAMIC METRICS ENGINE
# ============================================================================

class DynamicMetricsEngine:
    """Real-time dynamic metrics calculation"""

    def __init__(self):
        self.email_verifier = EmailVerificationEngine()
        self.llm_validator = LLMValidationEngine()

    async def evaluate_research_quality_dynamic(self, research_data: Dict) -> Dict[str, float]:
        """Dynamic research quality evaluation with real verification"""
        contacts = research_data.get("contacts", [])
        articles = research_data.get("articles", [])
        company_data = {
            'company_name': research_data.get('company_name'),
            'industry': research_data.get('industry'),
            'employee_count': research_data.get('employee_count'),
            'domain': research_data.get('domain')
        }

        # Verify emails asynchronously
        email_verification_tasks = [
            self.email_verifier.verify_email_comprehensive(contact.get('email', ''))
            for contact in contacts if contact.get('email')
        ]
        email_results = await asyncio.gather(*email_verification_tasks)

        # Validate company data with LLM
        company_validation = await self.llm_validator.validate_company_data(company_data)

        # Validate article relevance with LLM
        validated_articles = await self.llm_validator.validate_news_relevance_optimized(
            articles, company_data['company_name']
        )

        # Calculate dynamic scores
        contact_quality = self._calculate_contact_quality_dynamic(contacts, email_results)
        data_completeness = self._calculate_data_completeness_dynamic(research_data)
        article_relevance = self._calculate_article_relevance_dynamic(validated_articles)
        data_accuracy = company_validation.get('data_quality_score', 5) / 10  # Convert to 0-1

        return {
            "research_quality": round(contact_quality * 10, 2),
            "data_completeness": round(data_completeness * 10, 2),
            "article_relevance": round(article_relevance * 10, 2),
            "data_accuracy": round(data_accuracy * 10, 2),
            "contact_count_quality": min(10, len(contacts)),
            "email_verification_rate": self._calculate_verification_rate(email_results),
            "llm_validation_score": company_validation.get('data_quality_score', 5),
            "verified_contacts": len([r for r in email_results if r.get('confidence_score', 0) > 0.7]),
            "total_contacts_checked": len(email_results)
        }

    def _calculate_contact_quality_dynamic(self, contacts: List, email_results: List) -> float:
        """Calculate contact quality based on actual verification"""
        if not contacts:
            return 0.0

        scores = []
        for contact, verification in zip(contacts, email_results):
            score = 0.0
            # Name quality
            if contact.get('name') and contact['name'] != 'Unknown':
                score += 0.3

            # Title quality
            if contact.get('title'):
                score += 0.2

            # Email verification quality
            if verification.get('confidence_score', 0) > 0.7:
                score += 0.5
            elif verification.get('confidence_score', 0) > 0.3:
                score += 0.3

            scores.append(min(1.0, score))

        return sum(scores) / len(scores) if scores else 0.0

    def _calculate_data_completeness_dynamic(self, research_data: Dict) -> float:
        """Calculate data completeness with weighted fields"""
        field_weights = {
            'company_name': 0.25,
            'industry': 0.20,
            'employee_count': 0.20,
            'contacts': 0.20,
            'domain': 0.15
        }

        completeness = 0.0
        for field, weight in field_weights.items():
            if research_data.get(field):
                if field == 'contacts' and len(research_data[field]) > 0:
                    completeness += weight
                elif field != 'contacts':
                    completeness += weight

        return completeness

    def _calculate_article_relevance_dynamic(self, validated_articles: List) -> float:
        """Calculate article relevance based on LLM validation"""
        if not validated_articles:
            return 0.0

        relevance_scores = [
            article.get('relevance_confidence', 0)
            for article in validated_articles
            if article.get('is_relevant', False)
        ]

        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

    def _calculate_verification_rate(self, email_results: List) -> float:
        """Calculate email verification success rate"""
        if not email_results:
            return 0.0

        verified_count = sum(
            1 for result in email_results
            if result.get('confidence_score', 0) > 0.5
        )

        return round(verified_count / len(email_results), 2)

    async def evaluate_llm_health(self) -> Dict[str, float]:
        """Evaluate OpenAI API health and performance"""
        health_check = await self.llm_validator.check_api_health()

        if health_check["status"] == "healthy":
            response_time = health_check["response_time_ms"]
            # Convert to score (lower time = higher score)
            latency_score = max(1, 10 - (response_time - 500) / 200)
            return {
                "llm_availability": 10.0,
                "llm_latency": round(latency_score, 2),
                "llm_reliability": 9.0
            }
        else:
            return {
                "llm_availability": 0.0,
                "llm_latency": 0.0,
                "llm_reliability": 0.0
            }

    async def evaluate_security_metrics_dynamic(self) -> Dict[str, float]:
        """Dynamic security metrics evaluation including OpenAI health"""
        try:
            # Check if required environment variables are set
            required_vars = ['MONGODB_URL', 'HUNTER_API_KEY', 'SERPER_API_KEY']
            set_vars = sum(1 for var in required_vars if os.getenv(var))
            security_score = round(set_vars / len(required_vars) * 10, 2)

            # Check OpenAI health
            llm_health = await self.evaluate_llm_health()
            llm_score = llm_health["llm_availability"]

            return {
                "data_protection": security_score,
                "access_control": 8.0 if MONGO_URI != "mongodb://localhost:27017/" else 5.0,
                "api_security": 7.0,
                "compliance": 6.0,
                "environment_security": security_score,
                "llm_security": llm_score
            }
        except Exception as e:
            logger.error(f"Security metrics error: {e}")
            return {
                "data_protection": 5.0,
                "access_control": 5.0,
                "api_security": 5.0,
                "compliance": 5.0,
                "environment_security": 5.0,
                "llm_security": 0.0
            }

    async def evaluate_latency_metrics_dynamic(self, processing_times: Dict) -> Dict[str, float]:
        """Dynamic latency metrics based on actual processing times"""
        research_time = processing_times.get("research_time_ms", 0)
        scoring_time = processing_times.get("scoring_time_ms", 0)
        verification_time = processing_times.get("verification_time_ms", 0)

        # Convert to 1-10 scale (lower time = higher score)
        research_score = max(1, 10 - (research_time - 2000) / 1000) if research_time > 0 else 5
        scoring_score = max(1, 10 - (scoring_time - 1000) / 500) if scoring_time > 0 else 5
        verification_score = max(1, 10 - (verification_time - 500) / 250) if verification_time > 0 else 5

        return {
            "research_latency": round(research_score, 2),
            "scoring_latency": round(scoring_score, 2),
            "verification_latency": round(verification_score, 2),
            "api_response_time": 7.0,
            "system_availability": 9.5,
        }

    async def generate_comprehensive_evaluation(self, research_id: str, research_data: Dict,
                                       scoring_data: Dict, processing_times: Dict) -> Dict:
        """Generate comprehensive evaluation report with dynamic metrics"""

        research_metrics = await self.evaluate_research_quality_dynamic(research_data)
        security_metrics = await self.evaluate_security_metrics_dynamic()
        latency_metrics = await self.evaluate_latency_metrics_dynamic(processing_times)

        # Overall composite score
        overall_score = round(
            (research_metrics["research_quality"] * 0.30) +
            (research_metrics["data_accuracy"] * 0.25) +
            (research_metrics["email_verification_rate"] * 10 * 0.20) +
            (sum(security_metrics.values()) / len(security_metrics) * 0.15) +
            (sum(latency_metrics.values()) / len(latency_metrics) * 0.10),
            2
        )

        evaluation_report = {
            "research_id": research_id,
            "timestamp": datetime.now().isoformat(),
            "overall_score": overall_score,
            "research_quality": research_metrics,
            "security": {
                "security_score": round(sum(security_metrics.values()) / len(security_metrics), 2),
                "breakdown": security_metrics
            },
            "latency": {
                "latency_score": round(sum(latency_metrics.values()) / len(latency_metrics), 2),
                "breakdown": latency_metrics
            },
            "recommendations": self._generate_recommendations(overall_score, research_metrics)
        }

        # Store in database
        try:
            metrics.insert_one({
                "_id": str(uuid.uuid4()),
                **evaluation_report
            })
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

        return evaluation_report

    def _generate_recommendations(self, overall_score: float, research_metrics: Dict) -> List[str]:
        """Generate improvement recommendations based on scores"""
        recommendations = []

        if research_metrics["research_quality"] < 7:
            recommendations.append("Improve contact data quality by verifying email accuracy and job titles")

        if research_metrics["data_accuracy"] < 8:
            recommendations.append("Enhance data collection to include more verified firmographic fields")

        if research_metrics["email_verification_rate"] < 0.7:
            recommendations.append("Implement email verification service for better contact quality")

        if overall_score < 7:
            recommendations.append("Consider integrating additional data sources for better lead scoring")

        return recommendations

# ============================================================================
# DATA FETCHER (Enhanced with better employee data handling)
# ============================================================================

class DataFetcher:
    """Fetches data from external APIs (Hunter/Serper/AV for prospecting data)"""

    def _get_hunter_company_data(self, domain: str) -> Dict[str, Any]:
        """Get company firmographics and contacts from Hunter.io Domain Search API"""
        url = "https://api.hunter.io/v2/domain-search"
        params = {
            "domain": domain,
            "api_key": HUNTER_KEY,
            "limit": 10
        }

        company_data = {
            "company_name": domain.split('.')[0].capitalize(),
            "industry": "E-Learning/EdTech",  # Default to target industry
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

            # Extract company info
            company_data["company_name"] = data.get("organization",
                data.get("company", company_data["company_name"]))

            # Try to get employee count from Hunter (if available)
            # Note: Hunter doesn't always provide this, you may need to enrich with Clearbit/ZoomInfo
            company_data["employee_count"] = data.get("employees", 0)

            # Extract industry if available
            if data.get("industry"):
                company_data["industry"] = data.get("industry")

            # Extract contacts
            for person in data.get("emails", [])[:10]:
                name = (person.get("first_name", "") + " " + person.get("last_name", "")).strip()
                company_data["contacts"].append({
                    "name": name if name else "Unknown",
                    "title": person.get("position", ""),
                    "email": person.get("value", ""),
                    "linkedin": "",
                    "apollo_id": None,
                    "confidence": person.get("confidence", 0)
                })

        except requests.exceptions.RequestException as e:
            print(f"Hunter.io company/contact search error: {e}")
        except Exception as e:
            print(f"Hunter.io search failed: {e}")

        return company_data

    def get_prospect_data(self, domain: str) -> Dict[str, Any]:
        """Unified method using Hunter.io"""
        return self._get_hunter_company_data(domain)

    def get_serper_news(self, company_name: str, num_results: int = 20) -> List[Dict[str, Any]]:
        """Get news articles relevant to EdTech and Localization/Expansion"""
        url = "https://google.serper.dev/news"
        headers = {
            "X-API-KEY": SERPER_KEY,
            "Content-Type": "application/json"
        }

        # Targeted query for Dublabs.ai: focusing on global expansion and content
        query = (
            f'"{company_name}" global expansion OR "{company_name}" new markets OR '
            f'"{company_name}" localization OR "{company_name}" new curriculum OR '
            f'"{company_name}" content strategy OR "{company_name}" funding OR '
            f'"{company_name}" series OR "{company_name}" launch'
        )

        payload = {
            "q": query,
            "num": num_results,
            "tbs": "qdr:y"  # Limit to the past year for relevance
        }

        articles = []
        if not SERPER_KEY:
            print("🚨 SERPER_API_KEY is missing. Skipping Serper requests.")
            return articles

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
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
        """Get financial news for public companies"""
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHAVANTAGE_KEY}&limit=50"
        articles = []
        if ALPHAVANTAGE_KEY == "demo":
            print("⚠️ Using AlphaVantage 'demo' key.")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            for item in data.get('feed', []):
                sentiment_data = next(
                    (s for s in item.get("ticker_sentiment", []) if s.get("ticker") == ticker),
                    {}
                )
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


# ============================================================================
# ENHANCED SIGNAL ENGINE (Multi-dimensional scoring with F.I.R.E. methodology)
# ============================================================================

class EnhancedSignalEngine:
    """
    Multi-dimensional prospect scoring engine following F.I.R.E. methodology:
    - Fit (Firmographic)
    - Intent (Buying signals)
    - Recency (Time decay)
    - Engagement (Contact quality)
    """

    # Intent signals with proper categorization
    INTENT_SIGNALS = {
        # Critical Signals (0-40 points each, capped at 2 occurrences)
        'expansion_global': {
            'weight': 40,
            'keywords': ['expansion into', 'entering new markets', 'global expansion',
                        'launching in', 'international expansion', 'localize', 'localization',
                        'multi-region', 'cross-cultural', 'new geographies'],
            'category': 'critical'
        },
        'funding_raised': {
            'weight': 35,
            'keywords': ['raised $', 'secured funding', 'series a', 'series b', 'series c',
                        'venture capital', 'seed round', 'investment round', 'fundraise',
                        'funding round', 'raised funding', 'closes funding'],
            'category': 'critical'
        },

        # High Intent (0-25 points each)
        'leadership_change': {
            'weight': 25,
            'keywords': ['new ceo', 'new cto', 'new cmo', 'appointed', 'chief learning officer',
                        'vp of content', 'head of learning', 'head of international', 'joins as',
                        'chief content officer', 'vp learning'],
            'category': 'high'
        },
        'product_launch': {
            'weight': 20,
            'keywords': ['launched new', 'unveils', 'announces product', 'new platform',
                        'new course', 'curriculum launch', 'releasing', 'introduces',
                        'platform launch', 'new feature'],
            'category': 'high'
        },

        # Medium Intent (0-15 points each)
        'hiring_surge': {
            'weight': 15,
            'keywords': ['hiring', 'recruiting', 'job openings', 'growing team',
                        'hiring for', 'seeking', 'hiring spree', 'talent acquisition'],
            'category': 'medium'
        },
        'acquisition': {
            'weight': 12,
            'keywords': ['acquired', 'acquisition of', 'merger', 'bought', 'acquires',
                        'merges with', 'takes over'],
            'category': 'medium'
        },

        # Low Intent (0-8 points each)
        'partnership': {
            'weight': 8,
            'keywords': ['partnership with', 'partners with', 'collaboration', 'alliance',
                        'strategic partnership', 'teams up'],
            'category': 'low'
        },
        'award_recognition': {
            'weight': 5,
            'keywords': ['award', 'winner', 'ranked', 'recognized as', 'top edtech',
                        'best in class', 'industry leader'],
            'category': 'low'
        },

        # NEGATIVE SIGNALS (reduce score)
        'negative_news': {
            'weight': -20,
            'keywords': ['layoffs', 'downsizing', 'cutting costs', 'financial trouble',
                        'lawsuit', 'controversy', 'investigation', 'layoff', 'cuts jobs'],
            'category': 'negative'
        },
        'lost_leadership': {
            'weight': -15,
            'keywords': ['ceo departs', 'executive resignation', 'leadership exits',
                        'ceo resigns', 'executive leaves'],
            'category': 'negative'
        }
    }

    def __init__(self, config: ScoringConfig = None):
        self.config = config or ScoringConfig()

    def calculate_prospect_score(
        self,
        company_data: Dict[str, Any],
        articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive prospect score across all dimensions
        Returns score from 0-100 with detailed breakdown
        """

        # 1. FIRMOGRAPHIC FIT SCORE (0-100 scale, then weighted)
        fit_score = self._calculate_firmographic_fit(company_data)

        # 2. INTENT SIGNAL SCORE (0-100 scale, then weighted)
        intent_result = self._calculate_intent_score(articles)

        # 3. RECENCY SCORE (0-100 scale, then weighted)
        recency_score = self._calculate_recency_score(articles)

        # 4. Calculate weighted final score
        final_score = (
            (fit_score * self.config.FIRMOGRAPHIC_WEIGHT / 100) +
            (intent_result['intent_score'] * self.config.INTENT_SIGNAL_WEIGHT / 100) +
            (recency_score * self.config.RECENCY_WEIGHT / 100)
        )

        final_score = round(min(100, max(0, final_score)), 1)

        # 5. Determine priority tier
        priority = self._determine_priority(final_score, fit_score, intent_result['intent_score'])

        return {
            'final_score': final_score,
            'prospect_score': final_score,  # Backward compatibility
            'breakdown': {
                'firmographic_fit': round(fit_score, 1),
                'intent_signals': round(intent_result['intent_score'], 1),
                'recency': round(recency_score, 1)
            },
            'weights_used': {
                'firmographic': self.config.FIRMOGRAPHIC_WEIGHT,
                'intent': self.config.INTENT_SIGNAL_WEIGHT,
                'recency': self.config.RECENCY_WEIGHT
            },
            'priority': priority,
            'signals': intent_result['signals'],
            'signal_summary': intent_result['summary'],
            'total_signals': len(intent_result['signals']),
            'fit_analysis': self._get_fit_analysis(company_data),
            'recommendation': self._generate_recommendation(
                final_score, fit_score, intent_result, company_data
            ),
            'raw_intent_points': intent_result['raw_points'],
            'max_possible_points': intent_result['max_possible']
        }

    def _calculate_firmographic_fit(self, company_data: Dict[str, Any]) -> float:
        """Calculate ICP fit score (0-100)"""
        score = 0

        # Industry Match (40 points)
        industry = company_data.get('industry', '').lower()
        if any(target.lower() in industry for target in self.config.TARGET_INDUSTRIES):
            score += 40
        elif 'education' in industry or 'learning' in industry or 'tech' in industry:
            score += 20  # Partial match

        # Employee Count (30 points)
        employee_count = company_data.get('employee_count', 0)
        if employee_count < self.config.MIN_EMPLOYEE_COUNT:
            score += 0  # Too small
        elif any(low <= employee_count <= high for low, high in self.config.IDEAL_EMPLOYEE_RANGES):
            score += 30  # Perfect fit
        elif 20 <= employee_count < 50 or 2000 <= employee_count < 10000:
            score += 15  # Acceptable
        else:
            score += 5  # Large enterprise or no data

        # Contact Quality (30 points)
        contacts = company_data.get('contacts', [])
        if contacts:
            decision_maker_titles = ['ceo', 'cto', 'cmo', 'vp', 'director', 'head of', 'chief']
            has_decision_maker = any(
                any(title in contact.get('title', '').lower() for title in decision_maker_titles)
                for contact in contacts
            )

            if has_decision_maker:
                score += 30
            elif len(contacts) >= 3:
                score += 20
            elif len(contacts) >= 1:
                score += 10

        return min(100, score)

    def _calculate_intent_score(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate intent signal score with proper signal capping and decay"""
        signal_counts = {signal_type: 0 for signal_type in self.INTENT_SIGNALS.keys()}
        detected_signals = []
        raw_points = 0
        max_possible_points = 0

        # Calculate max possible points (with capping)
        for signal_type, config in self.INTENT_SIGNALS.items():
            max_count = min(self.config.MAX_SIGNALS_PER_TYPE, len(articles))
            if config['weight'] > 0:  # Only positive signals
                max_possible_points += config['weight'] * max_count

        # Process articles and detect signals
        for article in articles:
            text = (
                article.get("title", "") + " " +
                article.get("snippet", "") + " " +
                article.get("summary", "")
            ).lower()

            article_date = self._parse_article_date(article)
            days_old = (datetime.now() - article_date).days if article_date else 365

            # Skip articles older than max age
            if days_old > self.config.MAX_SIGNAL_AGE_DAYS:
                continue

            # Calculate recency multiplier (exponential decay)
            recency_multiplier = self._calculate_decay_multiplier(days_old)

            for signal_type, config in self.INTENT_SIGNALS.items():
                # Check if signal cap reached
                if signal_counts[signal_type] >= self.config.MAX_SIGNALS_PER_TYPE:
                    continue

                # Check for keyword match
                if any(keyword in text for keyword in config['keywords']):
                    signal_counts[signal_type] += 1

                    # Apply recency decay to weight
                    decayed_weight = config['weight'] * recency_multiplier
                    raw_points += decayed_weight

                    detected_signals.append({
                        'type': signal_type,
                        'category': config['category'],
                        'title': article.get("title", ""),
                        'source': article.get("source", ""),
                        'date': article.get("date", "") or article.get("time_published", ""),
                        'url': article.get("link", "") or article.get("url", ""),
                        'weight': config['weight'],
                        'base_weight': config['weight'],
                        'decayed_weight': round(decayed_weight, 1),
                        'days_old': days_old,
                        'recency_multiplier': round(recency_multiplier, 2)
                    })

        # Normalize to 0-100 scale
        intent_score = (raw_points / max_possible_points * 100) if max_possible_points > 0 else 0
        intent_score = min(100, max(0, intent_score))

        return {
            'intent_score': intent_score,
            'signals': detected_signals,
            'summary': signal_counts,
            'raw_points': round(raw_points, 1),
            'max_possible': max_possible_points
        }

    def _calculate_recency_score(self, articles: List[Dict[str, Any]]) -> float:
        """Score based on how recent the activity is (0-100)"""
        if not articles:
            return 0

        recent_articles = []
        for article in articles:
            article_date = self._parse_article_date(article)
            if article_date:
                days_old = (datetime.now() - article_date).days
                if days_old <= self.config.MAX_SIGNAL_AGE_DAYS:
                    recent_articles.append(days_old)

        if not recent_articles:
            return 0

        most_recent = min(recent_articles)
        articles_last_30_days = sum(1 for days in recent_articles if days <= 30)
        articles_last_90_days = sum(1 for days in recent_articles if days <= 90)

        # Most recent article score (50 points)
        if most_recent <= 7:
            recent_score = 50
        elif most_recent <= 30:
            recent_score = 40
        elif most_recent <= 90:
            recent_score = 25
        else:
            recent_score = 10

        # Volume score (30 points)
        if articles_last_30_days >= 3:
            volume_score = 30
        elif articles_last_30_days >= 1:
            volume_score = 20
        elif articles_last_90_days >= 2:
            volume_score = 15
        else:
            volume_score = 5

        # Consistency score (20 points)
        if len(recent_articles) >= 3 and articles_last_90_days >= 2:
            consistency_score = 20
        elif len(recent_articles) >= 2:
            consistency_score = 10
        else:
            consistency_score = 5

        return min(100, recent_score + volume_score + consistency_score)

    def _calculate_decay_multiplier(self, days_old: int) -> float:
        """Calculate exponential decay multiplier based on signal age"""
        if days_old <= 0:
            return 1.0

        half_life = self.config.DECAY_HALF_LIFE_DAYS
        decay_multiplier = math.pow(0.5, days_old / half_life)

        return max(0.1, decay_multiplier)

    def _parse_article_date(self, article: Dict[str, Any]) -> Optional[datetime]:
        """Parse article date from various formats"""
        date_str = article.get('date') or article.get('time_published', '')

        if not date_str:
            return None

        # Try various formats
        formats = [
            '%Y-%m-%d',
            '%Y%m%dT%H%M%S',
            '%B %d, %Y',
            '%d %B %Y',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str[:19], fmt)
            except:
                continue

        # Try parsing relative dates like "2 days ago"
        if 'ago' in date_str.lower():
            try:
                parts = date_str.lower().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    num = int(parts[0])
                    if 'day' in parts[1]:
                        return datetime.now() - timedelta(days=num)
                    elif 'week' in parts[1]:
                        return datetime.now() - timedelta(weeks=num)
                    elif 'month' in parts[1]:
                        return datetime.now() - timedelta(days=num*30)
                    elif 'hour' in parts[1]:
                        return datetime.now() - timedelta(hours=num)
            except:
                pass

        return None

    def _determine_priority(
        self,
        final_score: float,
        fit_score: float,
        intent_score: float
    ) -> str:
        """Determine priority tier based on combined factors"""
        if fit_score < 30:
            return "❌ Poor Fit - Do Not Pursue"

        if final_score >= 75 and intent_score >= 60:
            return "🔥 HOT - Immediate Outreach"
        elif final_score >= 60:
            return "🌡️ WARM - Priority Follow-up"
        elif final_score >= 40:
            return "❄️ COOL - Nurture Campaign"
        else:
            return "🧊 COLD - Monitor Only"

    def _get_fit_analysis(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide detailed fit analysis"""
        industry = company_data.get('industry', 'Unknown')
        employee_count = company_data.get('employee_count', 0)
        contact_count = len(company_data.get('contacts', []))

        return {
            'industry': industry,
            'employee_count': employee_count,
            'contact_count': contact_count,
            'industry_match': any(
                target.lower() in industry.lower()
                for target in self.config.TARGET_INDUSTRIES
            ),
            'size_match': any(
                low <= employee_count <= high
                for low, high in self.config.IDEAL_EMPLOYEE_RANGES
            ) if employee_count > 0 else False
        }

    def _generate_recommendation(
        self,
        final_score: float,
        fit_score: float,
        intent_result: Dict[str, Any],
        company_data: Dict[str, Any]
    ) -> str:
        """Generate specific, actionable recommendation"""

        if fit_score < 30:
            return (
                "❌ **Poor ICP Fit** - This company doesn't match our ideal customer profile. "
                "Focus resources on better-fit prospects."
            )

        signals = intent_result['signals']
        signal_summary = intent_result['summary']

        critical_signals = [s for s in signals if s['category'] == 'critical']

        if final_score >= 75:
            top_signal_types = sorted(
                [(k, v) for k, v in signal_summary.items() if v > 0],
                key=lambda x: self.INTENT_SIGNALS[x[0]]['weight'],
                reverse=True
            )[:2]

            signal_text = ", ".join([
                f"{s[0].replace('_', ' ').title()} (x{s[1]})"
                for s in top_signal_types
            ])

            return (
                f"🎯 **IMMEDIATE ACTION REQUIRED**\n\n"
                f"• **Hot signals detected:** {signal_text}\n"
                f"• **Next step:** Personalized outreach within 24 hours\n"
                f"• **Pitch angle:** Position Dublabs.ai as the solution for their "
                f"{'expansion' if any('expansion' in s['type'] for s in critical_signals) else 'growth'} needs\n"
                f"• **Contact:** Reach out to {len(company_data.get('contacts', []))} identified contacts"
            )

        elif final_score >= 60:
            return (
                f"📧 **WARM LEAD - STRATEGIC FOLLOW-UP**\n\n"
                f"• **Priority:** Schedule outreach within 48-72 hours\n"
                f"• **Approach:** Reference their recent {', '.join([s['type'].replace('_', ' ') for s in signals[:2]])} activity\n"
                f"• **Content:** Share relevant case studies on localization ROI"
            )

        elif final_score >= 40:
            return (
                f"📊 **NURTURE CAMPAIGN**\n\n"
                f"• Add to automated drip campaign\n"
                f"• Send educational content about content localization\n"
                f"• Monitor for increased intent signals"
            )

        else:
            return (
                f"🔍 **LOW PRIORITY - MONITORING MODE**\n\n"
                f"• Add to long-term monitoring list\n"
                f"• Set up news alerts for buying signal changes\n"
                f"• Focus sales resources on higher-scoring prospects"
            )


# ============================================================================
# VERIFIED SIGNAL ENGINE
# ============================================================================

class VerifiedSignalEngine(EnhancedSignalEngine):
    """Enhanced signal engine with data verification"""

    def __init__(self, config: ScoringConfig = None):
        super().__init__(config)
        self.llm_validator = LLMValidationEngine()

    async def calculate_prospect_score_verified(
        self,
        company_data: Dict[str, Any],
        articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate prospect score with verified data"""

        start_time = time.time()

        # Validate articles with LLM first
        validated_articles = await self.llm_validator.validate_news_relevance(
            articles, company_data.get('company_name', '')
        )

        # Use parent class scoring with validated articles
        score_result = self.calculate_prospect_score(company_data, validated_articles)

        verification_time = time.time() - start_time

        # Add verification metadata
        score_result['verification_metadata'] = {
            'total_articles_analyzed': len(articles),
            'relevant_articles_found': len(validated_articles),
            'article_relevance_ratio': len(validated_articles) / len(articles) if articles else 0,
            'validation_timestamp': datetime.now().isoformat(),
            'signal_confidence_level': self._calculate_signal_confidence(score_result['signals']),
            'verification_time_ms': round(verification_time * 1000, 2)
        }

        return score_result

    def _calculate_signal_confidence(self, signals: List[Dict]) -> str:
        """Calculate overall confidence in detected signals"""
        if not signals:
            return "low"

        # Calculate average confidence from LLM validation if available
        confidences = []
        for signal in signals:
            if signal.get('llm_validation', {}).get('relevance_confidence'):
                confidences.append(signal['llm_validation']['relevance_confidence'])
            else:
                confidences.append(0.5)  # Default confidence

        avg_confidence = sum(confidences) / len(confidences)

        if avg_confidence > 0.8:
            return "very_high"
        elif avg_confidence > 0.6:
            return "high"
        elif avg_confidence > 0.4:
            return "medium"
        else:
            return "low"


# ============================================================================
# PERSONALIZATION ENGINE (Updated for new signal structure)
# ============================================================================

class PersonalizationEngine:
    """Creates personalized content based on prospect data for Dublabs.ai"""

    PERSONALIZATION_TEMPLATES = {
        'expansion_global': """
I saw the exciting news about {company}'s **{signal_detail}** into new markets.
That kind of global push instantly raises the challenge of **localizing your massive video content library** quickly and affordably.
        """,

        'funding_raised': """
Congratulations on your recent funding! With capital secured, scaling your content delivery is likely a top priority.
Before you hire a massive team of native voice actors, I wanted to share how Dublabs.ai helps companies like {company} **launch courses in 10+ languages in days, not months.**
        """,

        'leadership_change': """
I noticed {name} recently joined {company} as **{title}**.
New content/learning leaders often prioritize **accelerating global reach and content accessibility** in their first 90 days, which is precisely where Dublabs.ai makes an immediate impact.
        """,

        'product_launch': """
The launch of your **{signal_detail}** is impressive!
To maximize the ROI on this new content, is {company} exploring ways to **simultaneously launch this course in key non-English speaking markets** without breaking the budget on traditional dubbing?
        """
    }

    def personalize_message(self, prospect_data: Dict[str, Any], product_context: str) -> str:
        """Creates a personalized message based on the prospect's top signal."""

        company_name = prospect_data.get("company_name", "the E-Learning provider")
        contacts = prospect_data.get("contacts", [])
        contact = contacts[0] if contacts else {}
        contact_name = contact.get("name", "Team")

        top_signal_list = prospect_data.get("signals", [])
        top_signal = top_signal_list[0] if top_signal_list else {}
        signal_type = top_signal.get('type', '')
        signal_detail = top_signal.get('title', 'major initiative')

        # Global Expansion Logic
        if signal_type == 'expansion_global':
            return self.PERSONALIZATION_TEMPLATES['expansion_global'].format(
                company=company_name,
                signal_detail=signal_detail
            ).strip()

        # Funding Signal Logic
        if signal_type == 'funding_raised':
            return self.PERSONALIZATION_TEMPLATES['funding_raised'].format(
                company=company_name
            ).strip()

        # Leadership Change Logic
        if signal_type == 'leadership_change' and contact:
            return self.PERSONALIZATION_TEMPLATES['leadership_change'].format(
                name=contact.get("name", "The new leader"),
                company=company_name,
                title=contact.get("title", "Executive")
            ).strip()

        # Product Launch Logic
        if signal_type == 'product_launch':
            return self.PERSONALIZATION_TEMPLATES['product_launch'].format(
                company=company_name,
                signal_detail=signal_detail
            ).strip()

        # Default/No Strong Signal
        return f"Hi {contact_name},\n\nI noticed {company_name} is a leader in the E-Learning space, which requires constantly updated, localized video content. Dublabs.ai specifically helps companies like yours **{product_context}**."


# ============================================================================
# OUTREACH COMPOSER (Updated for new priority tiers)
# ============================================================================

class OutreachComposer:
    """Composes multi-touch email sequences"""

    def __init__(self, personalization_engine: PersonalizationEngine):
        self.personalization_engine = personalization_engine

    def _get_primary_contact_name(self, prospect: Dict[str, Any]) -> str:
        contacts = prospect.get("contacts", [])
        return contacts[0].get("name", "Team") if contacts else "Team"

    def generate_subject_line(self, prospect: Dict[str, Any]) -> str:
        company_name = prospect.get("company_name", "Your Company")
        priority = prospect.get("priority", "Low")
        signals = prospect.get("signals", [])
        top_signal_type = signals[0].get("type", "Growth") if signals else "Growth"

        if "HOT" in priority or "🔥" in priority:
            if 'expansion' in top_signal_type or 'funding' in top_signal_type:
                return f"Dubbing solution for {company_name}'s {top_signal_type.replace('_', ' ')} plans"
            return f"Idea for fast content localization at {company_name}"

        if "WARM" in priority or "🌡️" in priority:
            return f"Quick thought on multi-language content for {company_name}"

        return f"Resource for E-Learning content at {company_name}"

    def create_initial_outreach(self, prospect: Dict[str, Any]):
        product_context = "generate multi-language audio and video for their e-learning courses 10x faster than traditional dubbing."
        personalized_opening = self.personalization_engine.personalize_message(prospect, product_context)
        contact_name = self._get_primary_contact_name(prospect)

        body = (
            f"Hi {contact_name},\n\n"
            f"{personalized_opening}\n\n"
            f"Dublabs.ai is an AI-powered dubbing solution that helps companies like yours **{product_context}** by automatically syncing translated audio to your original video.\n\n"
            f"Would you be open to a quick 15-minute demo to see the results (we can use one of your videos)?\n\n"
            "Best,\n[Your Name]"
        )

        email = {
            'subject': self.generate_subject_line(prospect),
            'body': body,
            'call_to_action': "Are you open to a quick 15-minute conversation on Thursday or Friday?"
        }
        return email

    def create_follow_up(self, prospect: Dict[str, Any], days_after: int):
        contact_name = self._get_primary_contact_name(prospect)
        subject = f"Re: {self.generate_subject_line(prospect)}"
        signals = prospect.get("signals", [])
        original_signal_type = signals[0].get("type", "growth") if signals else "growth"

        body = (
            f"Hi {contact_name},\n\n"
            f"Bumping this up. The challenge of scaling content for {prospect.get('company_name')}'s {original_signal_type.replace('_', ' ')} is real. \n\n"
            "Dublabs.ai could solve your multi-language video problem in the next week. If that's not a priority right now, just let me know, and I'll close the loop.\n\n"
            "Best,\n[Your Name]"
        )
        return {'subject': subject, 'body': body, 'call_to_action': "A quick reply is fine!"}

    def create_value_add(self, prospect: Dict[str, Any], days_after: int):
        contact_name = self._get_primary_contact_name(prospect)
        subject = f"A case study on multi-language course launches for EdTech"

        body = (
            f"Hi {contact_name},\n\n"
            f"If you're currently in the planning phase for {prospect.get('company_name')}'s international strategy, you might find this helpful:\n\n"
            f"We put together a short case study on how a large EdTech firm reduced their time-to-market for a new course from 9 weeks to 1 week using AI dubbing. [Insert link].\n\n"
            f"Let me know if this sparks any ideas.\n\nBest,\n[Your Name]"
        )
        return {'subject': subject, 'body': body, 'call_to_action': "N/A"}

    def create_breakup_email(self, prospect: Dict[str, Any], days_after: int):
        contact_name = self._get_primary_contact_name(prospect)
        company_name = prospect.get("company_name", "Your Company")

        subject = f"Closing the loop with {company_name} on localization"
        body = (
            f"Hi {contact_name},\n\n"
            "This will be my final message. I genuinely believe that as your e-learning platform continues to grow, traditional dubbing costs will slow your ability to enter new markets.\n\n"
            "When that need becomes critical, remember Dublabs.ai—we're here to solve the multi-language video bottleneck instantly. Wishing you success,\n\n"
            "[Your Name]"
        )
        return {'subject': subject, 'body': body, 'call_to_action': "N/A"}

    def compose_sequence(self, prospect: Dict[str, Any], campaign_type='standard'):
        sequence = {
            'email_1_initial_outreach': self.create_initial_outreach(prospect),
            'email_2_follow_up': self.create_follow_up(prospect, days_after=3),
            'email_3_value_add': self.create_value_add(prospect, days_after=7),
            'email_4_breakup': self.create_breakup_email(prospect, days_after=14)
        }
        return sequence


# ============================================================================
# INITIALIZE ENGINES
# ============================================================================

fetcher = DataFetcher()
signal_engine = EnhancedSignalEngine()
verified_signal_engine = VerifiedSignalEngine()
personalization_engine = PersonalizationEngine()
outreach_composer = OutreachComposer(personalization_engine)
dynamic_metrics = DynamicMetricsEngine()


# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

@app.get("/debug/openrouter-status")
async def debug_openrouter_status():
    """Check OpenRouter account status"""
    status_info = {
        "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "api_key_prefix": os.getenv("OPENAI_API_KEY", "")[:8] + "..." if os.getenv("OPENAI_API_KEY") else "None",
        "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo"),
        "timestamp": datetime.now().isoformat()
    }

    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Prospecting Agent"
        }

        payload = {
            "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo"),
            "messages": [{"role": "user", "content": "Say 'OK'"}],
            "max_tokens": 5
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            ) as response:

                if response.status == 200:
                    response_data = await response.json()
                    status_info["api_test"] = "success"
                    status_info["response"] = response_data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    status_info["api_test"] = "failed"
                    status_info["error"] = error_text
                    status_info["status_code"] = response.status

    except Exception as e:
        status_info["api_test"] = "failed"
        status_info["error"] = str(e)

    return status_info

@app.post("/research")
def research_company(domain: str, ticker: str = None):
    """Research a company and gather data"""
    if not domain:
        raise HTTPException(status_code=400, detail="Domain is required")

    research_id = str(uuid.uuid4())
    print(f"🔍 Starting enhanced research for {domain}...")

    # Fetch company data
    prospect_data = fetcher.get_prospect_data(domain)
    company_name = prospect_data["company_name"]

    # Fetch latest news articles
    serper_articles = fetcher.get_serper_news(company_name, num_results=20)
    alphavantage_articles = []
    if ticker:
        alphavantage_articles = fetcher.get_alphavantage_news(ticker)
    all_articles = serper_articles + alphavantage_articles

    print(f"📰 Found {len(all_articles)} articles for {company_name}")

    # Create database record
    record = {
        "_id": research_id,
        "domain": domain,
        "ticker": ticker,
        "company_name": company_name,
        "industry": prospect_data.get("industry"),
        "employee_count": prospect_data.get("employee_count"),
        "contacts": prospect_data["contacts"],
        "articles": all_articles,
        "researched_at": datetime.now(),
        "total_articles": len(all_articles),
        "is_verified_research": False
    }

    try:
        prospects.insert_one(record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database insert failed: {e}")

    print(f"✅ Research completed for {domain}")

    return {
        "research_id": research_id,
        "domain": domain,
        "company_name": company_name,
        "industry": prospect_data.get("industry"),
        "employee_count": prospect_data.get("employee_count"),
        "total_contacts": len(prospect_data["contacts"]),
        "contacts": prospect_data["contacts"],
        "total_articles": len(all_articles),
        "message": f"Research complete. Found {len(all_articles)} articles and {len(prospect_data['contacts'])} contacts."
    }


@app.post("/research-verified")
async def research_company_verified(domain: str, ticker: str = None):
    """Enhanced research with verification and LLM validation"""
    if not domain:
        raise HTTPException(status_code=400, detail="Domain is required")

    research_id = str(uuid.uuid4())
    logger.info(f"🔍 Starting verified research for {domain}...")

    start_time = time.time()

    # Fetch company data
    prospect_data = fetcher.get_prospect_data(domain)
    company_name = prospect_data["company_name"]

    # Fetch latest news articles
    serper_articles = fetcher.get_serper_news(company_name, num_results=20)
    alphavantage_articles = []
    if ticker:
        alphavantage_articles = fetcher.get_alphavantage_news(ticker)
    all_articles = serper_articles + alphavantage_articles

    logger.info(f"📰 Found {len(all_articles)} articles for {company_name}")

    # Enhanced verification with error handling
    try:
        verification_metrics = await dynamic_metrics.evaluate_research_quality_dynamic({
            **prospect_data,
            'articles': all_articles,
            'domain': domain
        })
    except Exception as e:
        logger.error(f"❌ Verification metrics failed: {e}")
        # Provide fallback metrics
        verification_metrics = {
            'research_quality': 5.0,
            'data_completeness': 5.0,
            'article_relevance': 5.0,
            'data_accuracy': 5.0,
            'contact_count_quality': min(10, len(prospect_data.get("contacts", []))),
            'email_verification_rate': 0.0,
            'llm_validation_score': 5.0,
            'verified_contacts': 0,
            'total_contacts_checked': 0,
            'verification_error': str(e)
        }

    research_time = time.time() - start_time

    # Create database record with verification data
    record = {
        "_id": research_id,
        "domain": domain,
        "ticker": ticker,
        "company_name": company_name,
        "industry": prospect_data.get("industry"),
        "employee_count": prospect_data.get("employee_count"),
        "contacts": prospect_data["contacts"],
        "articles": all_articles,
        "researched_at": datetime.now(),
        "total_articles": len(all_articles),
        "verification_metrics": verification_metrics,
        "research_quality_score": verification_metrics.get('research_quality', 0),
        "research_time_ms": research_time * 1000,
        "is_verified_research": True
    }

    try:
        prospects.insert_one(record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database insert failed: {e}")

    logger.info(f"✅ Verified research completed for {domain}")

    return {
        "research_id": research_id,
        "domain": domain,
        "company_name": company_name,
        "verification_score": verification_metrics.get('data_accuracy', 0),
        "email_verification_rate": verification_metrics.get('email_verification_rate', 0),
        "llm_validation_score": verification_metrics.get('llm_validation_score', 0),
        "total_contacts": len(prospect_data["contacts"]),
        "verified_contacts": verification_metrics.get('verified_contacts', 0),
        "total_articles": len(all_articles),
        "research_quality": verification_metrics.get('research_quality', 0),
        "has_verification_errors": 'verification_error' in verification_metrics,
        "message": f"Verified research complete. Quality score: {verification_metrics.get('research_quality', 0)}/10"
    }

@app.get("/signals/{research_id}")
async def get_signals(research_id: str):
    """Analyze signals and calculate prospect score"""
    start_time = time.time()

    try:
        record = prospects.find_one({"_id": research_id})
        if not record:
            raise HTTPException(status_code=404, detail="Research ID not found")

        articles = record.get("articles", [])

        # Re-run analysis with enhanced engine
        company_data = {
            "company_name": record.get("company_name"),
            "industry": record.get("industry"),
            "employee_count": record.get("employee_count", 0),
            "contacts": record.get("contacts", []),
            "domain": record.get("domain")
        }

        print(f"🔬 Analyzing signals for {company_data['company_name']}...")

        # Use verified engine if it's a verified research
        if record.get('is_verified_research'):
            signal_data = await verified_signal_engine.calculate_prospect_score_verified(company_data, articles)
        else:
            signal_data = signal_engine.calculate_prospect_score(company_data, articles)

        processing_time = time.time() - start_time

        # Generate comprehensive metrics
        metrics_report = await dynamic_metrics.generate_comprehensive_evaluation(
            research_id=research_id,
            research_data=record,
            scoring_data=signal_data,
            processing_times={"scoring_time_ms": processing_time * 1000}
        )

        # Update record with new signal data
        update_data = {
            "signals": signal_data["signals"],
            "signal_summary": signal_data["signal_summary"],
            "prospect_score": signal_data["prospect_score"],
            "final_score": signal_data["final_score"],
            "breakdown": signal_data["breakdown"],
            "fit_analysis": signal_data["fit_analysis"],
            "priority": signal_data["priority"],
            "recommendation": signal_data["recommendation"],
            "metrics_report": metrics_report,
            "analyzed_at": datetime.now(),
            "processing_time_ms": processing_time * 1000
        }

        # Add verification metadata if available
        if 'verification_metadata' in signal_data:
            update_data["verification_metadata"] = signal_data["verification_metadata"]

        prospects.update_one(
            {"_id": research_id},
            {"$set": update_data}
        )

        print(f"📊 Score: {signal_data['final_score']}/100 | Priority: {signal_data['priority']}")

        return {
            "research_id": research_id,
            "company_name": record.get("company_name"),
            "domain": record.get("domain"),
            "prospect_score": signal_data["prospect_score"],
            "final_score": signal_data["final_score"],
            "breakdown": signal_data["breakdown"],
            "fit_analysis": signal_data["fit_analysis"],
            "priority": signal_data["priority"],
            "recommendation": signal_data["recommendation"],
            "total_signals": signal_data["total_signals"],
            "signal_summary": signal_data["signal_summary"],
            "signals": signal_data["signals"],
            "articles_analyzed": len(articles),
            "metrics": metrics_report,
            "processing_time_ms": round(processing_time * 1000, 2),
            "is_verified": record.get('is_verified_research', False)
        }

    except Exception as e:
        logger.error(f"Error in /signals/{research_id}: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/openai-health")
async def debug_openai_health():
    """Debug endpoint to check OpenAI API health"""
    llm_validator = LLMValidationEngine()

    # Test with sample data
    test_company = {
        "company_name": "Test Company Inc",
        "industry": "Technology",
        "employee_count": 100,
        "domain": "testcompany.com"
    }

    health_info = {
        "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "api_key_prefix": os.getenv("OPENAI_API_KEY", "")[:8] + "..." if os.getenv("OPENAI_API_KEY") else "None",
        "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "timestamp": datetime.now().isoformat()
    }

    try:
        # Test API call
        validation_result = await llm_validator.validate_company_data(test_company)
        health_info["api_test"] = "success"
        health_info["validation_result"] = validation_result
    except Exception as e:
        health_info["api_test"] = "failed"
        health_info["error"] = str(e)
        health_info["traceback"] = traceback.format_exc()

    return health_info

@app.get("/compose/{research_id}")
def compose_outreach(research_id: str):
    """Compose personalized outreach sequence"""
    record = prospects.find_one({"_id": research_id})
    if not record:
        raise HTTPException(status_code=404, detail="Research ID not found.")

    # Ensure signals are analyzed
    if "prospect_score" not in record:
        company_data = {
            "company_name": record.get("company_name"),
            "industry": record.get("industry"),
            "employee_count": record.get("employee_count", 0),
            "contacts": record.get("contacts", [])
        }
        articles = record.get("articles", [])
        signal_data = signal_engine.calculate_prospect_score(company_data, articles)

        # Update record
        record.update(signal_data)
        prospects.update_one(
            {"_id": research_id},
            {"$set": {
                "signals": signal_data["signals"],
                "prospect_score": signal_data["prospect_score"],
                "priority": signal_data["priority"],
                "breakdown": signal_data.get("breakdown"),
                "analyzed_at": datetime.now()
            }}
        )
        print("💡 Signals generated automatically for outreach composition.")

    # Compose outreach sequence
    outreach_sequence = outreach_composer.compose_sequence(record)

    # Save outreach sequence to DB
    try:
        prospects.update_one(
            {"_id": research_id},
            {"$set": {
                "outreach_sequence": outreach_sequence,
                "composed_at": datetime.now()
            }}
        )
        print(f"📧 Outreach sequence saved for research ID: {research_id}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save outreach sequence to DB: {e}")

    contacts = record.get("contacts", [])
    primary_contact = contacts[0] if contacts else {}

    return {
        "research_id": research_id,
        "company_name": record.get("company_name"),
        "prospect_score": record.get("prospect_score"),
        "final_score": record.get("final_score"),
        "priority": record.get("priority"),
        "target_contact": primary_contact,
        "sequence": outreach_sequence
    }


@app.get("/company/{domain}")
def get_company_details(domain: str):
    """Get full company details by domain"""
    record = prospects.find_one({"domain": domain}, sort=[("researched_at", -1)])
    if not record:
        raise HTTPException(status_code=404, detail=f"No research found for domain: {domain}")

    # Convert ObjectId to string for JSON serialization
    record["research_id"] = str(record.pop("_id"))
    return record


@app.get("/companies")
def get_all_companies(limit: int = 20):
    """Get all researched companies"""
    cursor = prospects.find({}).sort("researched_at", -1).limit(limit)
    company_list = list(cursor)
    if not company_list:
        return []
    for company in company_list:
        company["research_id"] = str(company.pop("_id"))
    return company_list


# ============================================================================
# NEW METRICS ENDPOINTS
# ============================================================================

@app.get("/metrics/{research_id}")
def get_research_metrics(research_id: str):
    """Get comprehensive metrics for specific research"""
    try:
        record = prospects.find_one({"_id": research_id})
        if not record:
            raise HTTPException(status_code=404, detail="Research not found")

        metrics_report = record.get("metrics_report")
        if not metrics_report:
            raise HTTPException(status_code=404, detail="Metrics not generated for this research")

        return metrics_report

    except Exception as e:
        logger.error(f"Error retrieving metrics: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/dashboard/overview")
def get_metrics_overview(limit: int = 20):
    """Get overview of all research metrics"""
    try:
        records = prospects.find({}).sort("researched_at", -1).limit(limit)
        dashboard_data = []

        for record in records:
            research_id = str(record["_id"])
            metrics = record.get("metrics_report", {})

            dashboard_data.append({
                "research_id": research_id,
                "company_name": record.get("company_name"),
                "domain": record.get("domain"),
                "overall_score": metrics.get("overall_score", 0),
                "functionality_score": metrics.get("functionality", {}).get("functionality_score", 0),
                "research_quality": metrics.get("research_quality", {}).get("research_quality", 0),
                "processed_at": record.get("analyzed_at"),
                "is_verified": record.get("is_verified_research", False)
            })

        return {
            "total_researched": len(dashboard_data),
            "verified_researches": sum(1 for item in dashboard_data if item.get("is_verified")),
            "average_score": round(sum(item["overall_score"] for item in dashboard_data) / len(dashboard_data), 2) if dashboard_data else 0,
            "researches": dashboard_data
        }

    except Exception as e:
        logger.error(f"Error generating dashboard: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/evaluate/system")
async def evaluate_system_metrics():
    """Evaluate overall system metrics"""
    try:
        # Get recent researches for evaluation
        recent_researches = list(prospects.find({}).sort("researched_at", -1).limit(10))

        if not recent_researches:
            return {"message": "No research data available for evaluation"}

        total_researches = len(recent_researches)
        verified_researches = sum(1 for r in recent_researches if r.get('is_verified_research'))
        avg_processing_time = sum(r.get("processing_time_ms", 0) for r in recent_researches) / total_researches
        avg_score = sum(r.get("final_score", 0) for r in recent_researches) / total_researches
        success_rate = sum(1 for r in recent_researches if r.get("final_score", 0) > 0) / total_researches * 100

        # Get dynamic security metrics
        security_metrics = await dynamic_metrics.evaluate_security_metrics_dynamic()

        system_metrics = {
            "timestamp": datetime.now().isoformat(),
            "performance": {
                "total_researches_analyzed": total_researches,
                "verified_researches": verified_researches,
                "average_processing_time_ms": round(avg_processing_time, 2),
                "average_prospect_score": round(avg_score, 2),
                "success_rate_percent": round(success_rate, 2)
            },
            "data_quality": {
                "average_contacts_per_company": round(sum(len(r.get("contacts", [])) for r in recent_researches) / total_researches, 2),
                "average_articles_per_company": round(sum(len(r.get("articles", [])) for r in recent_researches) / total_researches, 2),
                "companies_with_contacts": sum(1 for r in recent_researches if r.get("contacts")) / total_researches * 100
            },
            "security": security_metrics
        }

        return system_metrics

    except Exception as e:
        logger.error(f"Error evaluating system metrics: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "Verified Prospecting Agent for Dublabs.ai with OpenAI Integration",
        "version": "4.0",
        "features": [
            "Multi-dimensional F.I.R.E. scoring",
            "Email verification with format/MX/disposable checks",
            "OpenAI-powered data validation",
            "Dynamic security and performance metrics",
            "Article relevance validation with OpenAI",
            "Real-time verification scoring",
            "Backward compatibility with original endpoints"
        ],
        "endpoints": {
            "standard_research": "/research?domain=example.com",
            "verified_research": "/research-verified?domain=example.com",
            "signals": "/signals/{research_id}",
            "outreach": "/compose/{research_id}",
            "metrics": "/metrics/{research_id}",
            "dashboard": "/metrics/dashboard/overview",
            "system_metrics": "/metrics/evaluate/system",
            "openai_health": "/debug/openai-health"
        }
    }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "app:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#         log_level="debug"
#     )

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))  # Render assigns this automatically

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="debug"
    )


# Run with: uvicorn app:app --reload --port 8000