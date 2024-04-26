from astropy.table import Table

from btk.match import PixelHungarianMatcher


def test_matching():
    x1 = [12.0, 31.0]
    y1 = [10.0, 30.0]
    x2 = [34.0, 12.1, 20.1]
    y2 = [33.0, 10.1, 22.0]

    t1 = Table()
    t1["x_peak"] = x1
    t1["y_peak"] = y1

    t2 = Table()
    t2["x_peak"] = x2
    t2["y_peak"] = y2

    catalog_list1 = [t1]
    catalog_list2 = [t2]

    matcher1 = PixelHungarianMatcher(pixel_max_sep=1)

    match = matcher1(catalog_list1, catalog_list2)

    assert match.n_true == 2
    assert match.n_pred == 3
    assert match.tp == 1
    assert match.fp == 2

    assert match.true_matches == [[0]]
    assert match.pred_matches == [[1]]
