import requests
import time
import xml.etree.ElementTree as ET
from typing import List, Optional
from loguru import logger
from pydantic import BaseModel
from src.config import settings


class PubMedArticle(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    publication_date: str
    journal: str
    doi: Optional[str] = None
    keywords: List[str] = []


class PubMedFetcher:
    """
    Fetches articles from PubMed using NCBI E-utilities API.
    Handles rate limiting, pagination, and XML parsing.

    Rate limits:
      Without API key : 3 requests/sec
      With API key    : 10 requests/sec (free at ncbiinsights.ncbi.nlm.nih.gov)
    """
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self):
        self.api_key = settings.NCBI_API_KEY
        self.delay = 0.1 if self.api_key else 0.35
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ClinicalAgent/1.0 (research)"})

    def search(self, query: str, max_results: int = 100) -> List[str]:
        """Search PubMed, return list of PMIDs."""
        logger.info(f"Searching PubMed: '{query}' | max={max_results}")

        params = {
            "db": "pubmed", "term": query,
            "retmax": max_results, "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        resp = self.session.get(f"{self.BASE_URL}esearch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()

        pmids = data["esearchresult"]["idlist"]
        total = data["esearchresult"]["count"]
        logger.info(f"Found {total} total | fetching {len(pmids)}")
        return pmids

    def fetch_articles(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch full article metadata for a list of PMIDs."""
        articles = []
        batch_size = settings.PUBMED_BATCH_SIZE

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            b_num = i // batch_size + 1
            b_total = (len(pmids) + batch_size - 1) // batch_size
            logger.info(f"Fetching batch {b_num}/{b_total}")

            params = {
                "db": "pubmed", "id": ",".join(batch),
                "rettype": "xml", "retmode": "xml",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            resp = self.session.get(f"{self.BASE_URL}efetch.fcgi", params=params)
            resp.raise_for_status()
            articles.extend(self._parse_xml(resp.text))
            time.sleep(self.delay)

        logger.success(f"Fetched {len(articles)} articles")
        return articles

    def _parse_xml(self, xml_text: str) -> List[PubMedArticle]:
        articles = []
        root = ET.fromstring(xml_text)
        for elem in root.findall(".//PubmedArticle"):
            try:
                a = self._parse_single(elem)
                if a:
                    articles.append(a)
            except Exception as e:
                logger.warning(f"Skipping malformed article: {e}")
        return articles

    def _parse_single(self, elem) -> Optional[PubMedArticle]:
        pmid_elem = elem.find(".//PMID")
        if pmid_elem is None:
            return None

        # Handle structured abstracts (Background / Methods / Results sections)
        abstract_parts = elem.findall(".//AbstractText")
        abstract = " ".join([
            f"{p.get('Label')}: {p.text or ''}" if p.get('Label')
            else (p.text or "")
            for p in abstract_parts
        ]).strip()

        if not abstract:
            return None  # Skip articles with no abstract

        authors = []
        for author in elem.findall(".//Author"):
            last = author.findtext("LastName", "")
            first = author.findtext("ForeName", "")
            name = f"{first} {last}".strip() if first else last
            if name:
                authors.append(name)

        pub_date = elem.find(".//PubDate")
        year = pub_date.findtext("Year", "") if pub_date is not None else ""
        month = pub_date.findtext("Month", "") if pub_date is not None else ""

        doi = next(
            (e.text for e in elem.findall(".//ArticleId")
             if e.get("IdType") == "doi"), None
        )

        return PubMedArticle(
            pmid=pmid_elem.text,
            title=(elem.findtext(".//ArticleTitle") or "Unknown").strip(),
            abstract=abstract,
            authors=authors,
            publication_date=f"{year}-{month}".strip("-"),
            journal=elem.findtext(".//Journal/Title", "Unknown Journal"),
            doi=doi,
            keywords=[kw.text for kw in elem.findall(".//Keyword") if kw.text],
        )

    def search_and_fetch(self, query: str, max_results: int = 100) -> List[PubMedArticle]:
        pmids = self.search(query, max_results)
        return self.fetch_articles(pmids)
