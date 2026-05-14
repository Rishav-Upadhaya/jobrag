"""
Query parsing utilities for extracting parameters from user queries.
"""

import re

MAX_TOP_K = 50

_LOCATION_ALIASES = {
	"nyc": "New York",
	"new york city": "New York",
	"sf": "San Francisco",
	"bay area": "San Francisco",
}


def extract_top_k_from_query(query: str) -> int | None:
	"""
	Extract a top_k value from a user query.
	
	Looks for patterns like:
	- "top 20", "top 20 results"
	- "need 20", "need 20 results"
	- "show 20", "show me 20"
	- "give me 20"
	- "I want 20"
	- "get 20"
	- "fetch 20"
	- "retrieve 20"
	- "return 20"
	- "find 20" (but not "find 20 senior" - needs "results" or "jobs")
	
	Args:
		query: User query string
		
	Returns:
		Extracted top_k value (1-50), or None if not found or invalid
	"""
	if not query:
		return None
	
	# Patterns that explicitly request a number of results
	patterns = [
		r'(?:top|show|give|get|fetch|retrieve|return|need)\s+(?:me\s+)?(\d+)',
		r'(?:I\s+)?want\s+(\d+)',
		r'(\d+)\s+results',
		r'(\d+)\s+jobs',
	]
	
	for pattern in patterns:
		match = re.search(pattern, query, re.IGNORECASE)
		if match:
			try:
				value = int(match.group(1))
				if value >= 1:
					return min(value, MAX_TOP_K)
			except (ValueError, IndexError):
				pass
	
	return None


def extract_filters_from_query(query: str) -> dict[str, str]:
	"""Extract conservative structured filters from common job-search phrasing."""
	if not query:
		return {}

	lowered = query.lower()
	filters: dict[str, str] = {}

	level_patterns = [
		("Intern", r"\b(intern|internship)\b"),
		("Entry", r"\b(entry[- ]level|graduate|new grad)\b"),
		("Junior", r"\b(junior|jr\.?)\b"),
		("Mid", r"\b(mid[- ]level|intermediate)\b"),
		("Senior", r"\b(senior|sr\.?)\b"),
		("Lead", r"\b(lead|principal|staff)\b"),
	]
	for level, pattern in level_patterns:
		if re.search(pattern, lowered):
			filters["job_level"] = level
			break

	category_patterns = [
		("Engineering", r"\b(engineering|software|developer|devops|backend|frontend|full[- ]stack|ml|machine learning|ai)\b"),
		("Data", r"\b(data|analytics|analyst|scientist|bi|business intelligence)\b"),
		("Product", r"\b(product|pm|product manager)\b"),
		("Design", r"\b(design|designer|ux|ui)\b"),
		("Marketing", r"\b(marketing|growth|seo|content)\b"),
		("Sales", r"\b(sales|account executive|business development|bd)\b"),
		("Operations", r"\b(operations|ops|support|customer success|admin)\b"),
	]
	for category, pattern in category_patterns:
		if re.search(pattern, lowered):
			filters["job_category"] = category
			break

	location_match = re.search(
		r"\b(?:in|near|around|at)\s+([a-z][a-z .'-]{1,40})(?:\s+(?:for|with|at|that|which|who|and|or)\b|[,.;!?]|$)",
		query,
		re.IGNORECASE,
	)
	if location_match:
		location = " ".join(location_match.group(1).strip().split())
		location = _LOCATION_ALIASES.get(location.lower(), location)
		if location and not re.fullmatch(r"(remote|hybrid|onsite|office)", location, re.IGNORECASE):
			filters["job_location"] = location
	elif re.search(r"\bremote\b", lowered):
		filters["job_location"] = "Remote"

	return filters


def is_generic_job_query(query: str) -> bool:
	"""Return true for job-search requests that lack a searchable role or attribute."""
	if not query:
		return False
	lowered = query.lower()
	if not re.search(r"\b(job|jobs|role|roles|opening|openings|position|positions)\b", lowered):
		return False
	specific_terms = [
		"engineer", "developer", "manager", "analyst", "specialist", "intern",
		"lead", "senior", "junior", "data", "software", "ml", "machine learning",
		"ai", "architect", "scientist", "consultant", "executive", "director",
		"coordinator", "support", "technician", "admin", "designer", "product",
		"marketing", "sales", "backend", "frontend", "devops",
	]
	return not any(term in lowered for term in specific_terms)


def is_similar_role_followup(query: str) -> bool:
	"""Detect follow-ups that should inherit the previous role/search query."""
	if not query:
		return False
	return bool(
		re.search(
			r"\b(similar|same|related|like\s+that|like\s+this|those|them)\b.*\b(job|jobs|role|roles|position|positions|opening|openings)\b"
			r"|\b(job|jobs|role|roles|position|positions|opening|openings)\b.*\b(similar|same|related|like\s+that|like\s+this|those|them)\b",
			query,
			re.IGNORECASE,
		)
	)
