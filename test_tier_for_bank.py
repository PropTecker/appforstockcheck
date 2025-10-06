"""
Test suite for tier_for_bank function to verify BNG "either-axis, pick the best" rule.

This test verifies that:
1. LPA and NCA are evaluated independently
2. The best (closest) category is returned
3. No cross-layer comparisons are made
4. Missing values on one axis don't prevent the other axis from working
"""

import re
from typing import List, Optional

# Copied from app.py to avoid importing streamlit dependencies
def sstr(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        import math
        if math.isnan(x) or math.isinf(x):
            return ""
    return str(x).strip()

def norm_name(s: str) -> str:
    t = sstr(s).lower()
    t = re.sub(r'\b(city of|royal borough of|metropolitan borough of)\b', '', t)
    t = re.sub(r'\b(council|borough|district|county|unitary authority|unitary|city)\b', '', t)
    t = t.replace("&", "and")
    t = re.sub(r'[^a-z0-9]+', '', t)
    return t

def tier_for_bank(bank_lpa: str, bank_nca: str,
                  t_lpa: str, t_nca: str,
                  lpa_neigh: List[str], nca_neigh: List[str],
                  lpa_neigh_norm: Optional[List[str]] = None,
                  nca_neigh_norm: Optional[List[str]] = None) -> str:
    b_lpa = norm_name(bank_lpa)
    b_nca = norm_name(bank_nca)
    t_lpa_n = norm_name(t_lpa)
    t_nca_n = norm_name(t_nca)
    if lpa_neigh_norm is None:
        lpa_neigh_norm = [norm_name(x) for x in (lpa_neigh or [])]
    if nca_neigh_norm is None:
        nca_neigh_norm = [norm_name(x) for x in (nca_neigh or [])]
    if b_lpa and t_lpa_n and b_lpa == t_lpa_n:
        return "local"
    if b_nca and t_nca_n and b_nca == t_nca_n:
        return "local"
    if b_lpa and b_lpa in lpa_neigh_norm:
        return "adjacent"
    if b_nca and b_nca in nca_neigh_norm:
        return "adjacent"
    return "far"


def test_same_lpa_different_nca_returns_local():
    """Same LPA, different NCA → local (LPA axis wins)"""
    result = tier_for_bank(
        bank_lpa="Westminster",
        bank_nca="Thames Valley",
        t_lpa="Westminster",
        t_nca="Chilterns",
        lpa_neigh=[],
        nca_neigh=[]
    )
    assert result == "local", f"Expected 'local' but got '{result}'"
    print("✓ Same LPA, different NCA → local")


def test_neighbour_lpa_unrelated_nca_returns_adjacent():
    """Neighbour LPA, unrelated NCA → adjacent (LPA axis wins)"""
    result = tier_for_bank(
        bank_lpa="Camden",
        bank_nca="Thames Valley",
        t_lpa="Westminster",
        t_nca="Chilterns",
        lpa_neigh=["Camden", "Kensington"],
        nca_neigh=["South Downs"]
    )
    assert result == "adjacent", f"Expected 'adjacent' but got '{result}'"
    print("✓ Neighbour LPA, unrelated NCA → adjacent")


def test_same_nca_different_lpa_returns_local():
    """Same NCA, different LPA → local (NCA axis wins)"""
    result = tier_for_bank(
        bank_lpa="Camden",
        bank_nca="Thames Valley",
        t_lpa="Westminster",
        t_nca="Thames Valley",
        lpa_neigh=[],
        nca_neigh=[]
    )
    assert result == "local", f"Expected 'local' but got '{result}'"
    print("✓ Same NCA, different LPA → local")


def test_neighbour_nca_unrelated_lpa_returns_adjacent():
    """Neighbour NCA, unrelated LPA → adjacent (NCA axis wins)"""
    result = tier_for_bank(
        bank_lpa="Manchester",
        bank_nca="South Downs",
        t_lpa="Westminster",
        t_nca="Thames Valley",
        lpa_neigh=["Camden"],
        nca_neigh=["South Downs", "Chilterns"]
    )
    assert result == "adjacent", f"Expected 'adjacent' but got '{result}'"
    print("✓ Neighbour NCA, unrelated LPA → adjacent")


def test_different_and_nonneighbour_returns_far():
    """Different & non-neighbour on both axes → far"""
    result = tier_for_bank(
        bank_lpa="Manchester",
        bank_nca="Peak District",
        t_lpa="Westminster",
        t_nca="Thames Valley",
        lpa_neigh=["Camden"],
        nca_neigh=["South Downs"]
    )
    assert result == "far", f"Expected 'far' but got '{result}'"
    print("✓ Different & non-neighbour on both axes → far")


def test_missing_nca_same_lpa_returns_local():
    """Missing NCA for bank, same LPA → local (LPA axis still works)"""
    result = tier_for_bank(
        bank_lpa="Westminster",
        bank_nca="",
        t_lpa="Westminster",
        t_nca="Thames Valley",
        lpa_neigh=[],
        nca_neigh=[]
    )
    assert result == "local", f"Expected 'local' but got '{result}'"
    print("✓ Missing NCA for bank, same LPA → local")


def test_missing_lpa_neighbour_nca_returns_adjacent():
    """Missing LPA for bank, neighbour NCA → adjacent (NCA axis still works)"""
    result = tier_for_bank(
        bank_lpa="",
        bank_nca="South Downs",
        t_lpa="Westminster",
        t_nca="Thames Valley",
        lpa_neigh=["Camden"],
        nca_neigh=["South Downs"]
    )
    assert result == "adjacent", f"Expected 'adjacent' but got '{result}'"
    print("✓ Missing LPA for bank, neighbour NCA → adjacent")


def test_normalisation_applied():
    """Verify normalization is applied to names before comparison"""
    # Test with variations in capitalization and extra words
    result = tier_for_bank(
        bank_lpa="City of Westminster Borough Council",
        bank_nca="Thames Valley",
        t_lpa="Westminster",
        t_nca="Thames Valley",
        lpa_neigh=[],
        nca_neigh=[]
    )
    assert result == "local", f"Expected 'local' but got '{result}'"
    print("✓ Normalization applied correctly")


def test_uses_normalized_neighbour_lists():
    """Verify that pre-normalized neighbour lists are used if provided"""
    result = tier_for_bank(
        bank_lpa="Camden Borough",
        bank_nca="Thames Valley",
        t_lpa="Westminster",
        t_nca="Chilterns",
        lpa_neigh=["Camden Borough Council", "Kensington & Chelsea"],
        nca_neigh=[],
        lpa_neigh_norm=[norm_name("Camden Borough Council"), norm_name("Kensington & Chelsea")],
        nca_neigh_norm=[]
    )
    assert result == "adjacent", f"Expected 'adjacent' but got '{result}'"
    print("✓ Pre-normalized neighbour lists used correctly")


def test_empty_neighbour_lists():
    """If neighbour lists are empty, only same/outside categories possible"""
    # Same LPA with empty lists
    result = tier_for_bank(
        bank_lpa="Westminster",
        bank_nca="Peak District",
        t_lpa="Westminster",
        t_nca="Thames Valley",
        lpa_neigh=[],
        nca_neigh=[]
    )
    assert result == "local", f"Expected 'local' but got '{result}'"
    
    # Different on both axes with empty lists
    result = tier_for_bank(
        bank_lpa="Manchester",
        bank_nca="Peak District",
        t_lpa="Westminster",
        t_nca="Thames Valley",
        lpa_neigh=[],
        nca_neigh=[]
    )
    assert result == "far", f"Expected 'far' but got '{result}'"
    print("✓ Empty neighbour lists handled correctly")


def test_both_axes_local_returns_local():
    """When both axes are local, should return local"""
    result = tier_for_bank(
        bank_lpa="Westminster",
        bank_nca="Thames Valley",
        t_lpa="Westminster",
        t_nca="Thames Valley",
        lpa_neigh=[],
        nca_neigh=[]
    )
    assert result == "local", f"Expected 'local' but got '{result}'"
    print("✓ Both axes local → local")


def test_lpa_local_nca_adjacent_returns_local():
    """When LPA is local and NCA is adjacent, should return local (best wins)"""
    result = tier_for_bank(
        bank_lpa="Westminster",
        bank_nca="South Downs",
        t_lpa="Westminster",
        t_nca="Thames Valley",
        lpa_neigh=[],
        nca_neigh=["South Downs"]
    )
    assert result == "local", f"Expected 'local' but got '{result}'"
    print("✓ LPA local, NCA adjacent → local (best wins)")


def test_lpa_adjacent_nca_far_returns_adjacent():
    """When LPA is adjacent and NCA is far, should return adjacent (best wins)"""
    result = tier_for_bank(
        bank_lpa="Camden",
        bank_nca="Peak District",
        t_lpa="Westminster",
        t_nca="Thames Valley",
        lpa_neigh=["Camden"],
        nca_neigh=[]
    )
    assert result == "adjacent", f"Expected 'adjacent' but got '{result}'"
    print("✓ LPA adjacent, NCA far → adjacent (best wins)")


if __name__ == "__main__":
    print("Running tier_for_bank tests...\n")
    
    test_same_lpa_different_nca_returns_local()
    test_neighbour_lpa_unrelated_nca_returns_adjacent()
    test_same_nca_different_lpa_returns_local()
    test_neighbour_nca_unrelated_lpa_returns_adjacent()
    test_different_and_nonneighbour_returns_far()
    test_missing_nca_same_lpa_returns_local()
    test_missing_lpa_neighbour_nca_returns_adjacent()
    test_normalisation_applied()
    test_uses_normalized_neighbour_lists()
    test_empty_neighbour_lists()
    test_both_axes_local_returns_local()
    test_lpa_local_nca_adjacent_returns_local()
    test_lpa_adjacent_nca_far_returns_adjacent()
    
    print("\n✅ All tests passed!")
