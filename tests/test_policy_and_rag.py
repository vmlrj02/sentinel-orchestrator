import pytest

from sentinel.tenants import TENANTS
from tests.conftest import retriever

APP = {"loan_amnt": 10_000, "person_income": 50_000}


@pytest.mark.parametrize(("tenant", "probability", "application", "expected"), [
    ("aster", 0.10, APP, "approved"),
    ("aster", 0.45, APP, "referred"),        # review band 30-60%
    ("aster", 0.75, APP, "declined"),
    ("aster", 0.10, {"loan_amnt": 25_000, "person_income": 100_000}, "referred"),  # above auto limit
    ("aster", 0.10, {"loan_amnt": 25_000, "person_income": 50_000}, "declined"),   # 50% > 40% LTI cap
    ("kestrel", 0.10, {"loan_amnt": 25_000, "person_income": 50_000}, "referred"),  # 50% OK, above 15k limit
    ("kestrel", 0.35, APP, "approved"),      # Kestrel approves below 40%
])
def test_policy_decisions(tenant, probability, application, expected):
    assert TENANTS[tenant].decide(probability, application)[0] == expected


def test_policy_thresholds_match_policy_documents():
    """The rules in code and the documents the advisor cites must say the same thing."""
    aster = " ".join(c.text for c in retriever().chunks if c.tenant_id == "aster")
    kestrel = " ".join(c.text for c in retriever().chunks if c.tenant_id == "kestrel")
    assert "below 30%" in aster and "above 60%" in aster and "40% of your gross annual income" in aster
    assert "below 40%" in kestrel and "above 70%" in kestrel and "50% of your gross annual income" in kestrel
    assert "$20,000" in aster and "$15,000" in kestrel


def test_retrieval_is_tenant_scoped():
    for tenant in ("aster", "kestrel"):
        results = retriever().search("origination fee early repayment", tenant)
        assert results and all(r.chunk.tenant_id == tenant for r in results)
    assert retriever().search("fees", "unknown-lender") == []


def test_keyword_queries_find_the_right_section():
    top = retriever().search("appeal a declined application", "kestrel")[0]
    assert top.chunk.id == "kestrel/04-declines-appeals-improving#appeals"


def test_unretrieved_citations_are_removed():
    from langchain_core.messages import ToolMessage

    from sentinel.agent.graph import strip_unretrieved_citations

    retrieved = ToolMessage("[kestrel/03-rates-fees-repayment#fees]\nNo origination fee.", tool_call_id="1",
                            name="search_policy")
    text = "No fee [kestrel/03-rates-fees-repayment#fees]. Approved [kestrel/decision-record]."
    assert strip_unretrieved_citations(text, [retrieved]) == \
        "No fee [kestrel/03-rates-fees-repayment#fees]. Approved."
