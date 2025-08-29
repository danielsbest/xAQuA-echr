from scipy import stats
from scipy.stats import rankdata


def normalize_scores(scores: list[float]) -> list[float]:
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return scores
    return [(score - min_score) / (max_score - min_score) for score in scores]


def get_ranks(auto: list[tuple[str, float]], expert: list[str]):
    try:
        auto = sorted(auto, key=lambda x: expert.index(x[0]))  # sort by expert order
        scores = normalize_scores([score for _, score in auto])  # normalize scores
        auto_ranks = rankdata(scores, method="average")  # get ranks
        auto_ranks = len(auto_ranks) + 1 - auto_ranks  # reverse ranks
        expert_ranks = [i + 1 for i in range(len(expert))]  # get expert ranks
        return auto_ranks.tolist(), expert_ranks
    except Exception as e:
        print("Auto:", auto)
        print("Expert:", expert)
        raise e from e


def spearmanr(auto: list[list[tuple[str, float]]], expert: list[list[str]]):
    auto_full_list = []
    expert_full_list = []
    for auto_group, expert_group in zip(auto, expert):
        auto_ranks, expert_ranks = get_ranks(auto_group, expert_group)
        auto_full_list.extend(auto_ranks)
        expert_full_list.extend(expert_ranks)

    correlation, p_value = stats.spearmanr(auto_full_list, expert_full_list)
    return correlation, p_value


def test_get_ranks():
    expert = ["a", "b", "c"]
    auto = [("a", 3), ("b", 2), ("c", 1)]
    auto_ranks, expert_ranks = get_ranks(auto, expert)
    assert expert_ranks == [1, 2, 3]
    assert auto_ranks == [1, 2, 3]

    expert = ["a", "b", "c"]
    auto = [("b", 2), ("a", 1), ("c", 2)]
    auto_ranks, expert_ranks = get_ranks(auto, expert)
    assert expert_ranks == [1, 2, 3]
    assert auto_ranks == [3, 1.5, 1.5]

    expert = ["c", "b", "a"]
    auto = [("b", 0.33), ("a", 0.33), ("c", 0.5)]
    auto_ranks, expert_ranks = get_ranks(auto, expert)
    assert expert_ranks == [1, 2, 3]
    assert auto_ranks == [1, 2.5, 2.5]

    expert = ["a", "b", "c"]
    auto = [("a", 0.33), ("b", 0.33), ("c", 0.33)]
    auto_ranks, expert_ranks = get_ranks(auto, expert)
    assert expert_ranks == [1, 2, 3]
    assert auto_ranks == [2, 2, 2]

    expert = ["b", "a", "c"]
    auto = [("a", 2), ("b", 3), ("c", 1)]
    auto_ranks, expert_ranks = get_ranks(auto, expert)
    assert expert_ranks == [1, 2, 3]
    assert auto_ranks == [1, 2, 3]


def main():
    test_get_ranks()


if __name__ == "__main__":
    main()
