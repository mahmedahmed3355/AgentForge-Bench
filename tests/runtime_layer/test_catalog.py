import pytest

from agentforge.runtime.catalog import TaskCatalog, TaskDescriptor


def test_add_and_get_descriptor():
    catalog = TaskCatalog()

    descriptor = TaskDescriptor(
        task_id="cuda-001",
        domain="cuda",
        name="kernel-indexing",
    )

    catalog.add(descriptor)

    assert catalog.get("cuda-001") == descriptor
    assert catalog.contains("cuda-001")


def test_duplicate_descriptor_rejected():
    catalog = TaskCatalog()
    descriptor = TaskDescriptor(
        task_id="cuda-001",
        domain="cuda",
        name="kernel-indexing",
    )

    catalog.add(descriptor)

    with pytest.raises(ValueError):
        catalog.add(descriptor)


def test_catalog_order_is_deterministic():
    catalog = TaskCatalog()

    catalog.add(
        TaskDescriptor(
            task_id="b",
            domain="cuda",
            name="b",
        )
    )
    catalog.add(
        TaskDescriptor(
            task_id="a",
            domain="cuda",
            name="a",
        )
    )

    assert tuple(
        item.task_id
        for item in catalog.all()
    ) == ("a", "b")
