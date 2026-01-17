import torch

from pname.data import arxiv_dataset


def test_data():
    """Test that arxiv_dataset loads correctly."""
    train, test = arxiv_dataset()

    # Check that datasets are not empty
    assert len(train) > 0, "Training dataset should not be empty"
    assert len(test) > 0, "Test dataset should not be empty"

    # Check that each dataset returns (text, label) tuples
    for dataset in [train, test]:
        text, label = dataset[0]
        assert isinstance(text, str), "First element should be text (string)"
        assert isinstance(label, torch.Tensor), "Second element should be label (tensor)"
        assert label.dtype == torch.long, "Label should be long tensor"
        assert len(text) > 0, "Text should not be empty"

    # Check that labels are valid (non-negative integers)
    train_labels = torch.stack([train[i][1] for i in range(min(100, len(train)))])
    test_labels = torch.stack([test[i][1] for i in range(min(100, len(test)))])

    assert (train_labels >= 0).all(), "All training labels should be non-negative"
    assert (test_labels >= 0).all(), "All test labels should be non-negative"
