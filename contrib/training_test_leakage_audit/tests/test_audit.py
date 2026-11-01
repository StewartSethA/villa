from contrib.training_test_leakage_audit.leakage_audit import audit, seg_tokens


def test_token_matches_representation_names_that_point_at_a_test_segment():
    train = [dict(name="pherc0139-w017", physical_segment="20250108000005-w030_2025010818"), dict(name="pherc0139-w016", physical_segment="20250108000004-w029_2025010827"),
             dict(name="20260126000000-w045_2026012619"), dict(name="pherc0814-46527", physical_segment="20260226000000-46527_2um_try2")]
    test = ["20250108000005", "20260126000000", "46527_2um_try2"]
    hits = audit(train, test)
    assert [h["train_name"] for h in hits] == ["pherc0139-w017", "20260126000000-w045_2026012619", "pherc0814-46527"]
    assert "20260220213127" in seg_tokens("PHerc0841__20260220213127-w00")


def test_no_overlap_gives_no_hits():
    assert audit([dict(name="a", physical_segment="20250101000000-x")], ["20250202000000"]) == []
