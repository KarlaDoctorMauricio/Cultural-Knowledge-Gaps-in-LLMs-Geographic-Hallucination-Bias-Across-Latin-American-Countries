from fairness_toolkit.progress import count_response_tasks, task_progress


def test_count_response_tasks_without_resume():
    import pandas as pd

    from fairness_toolkit.checkpoint import response_is_done

    df = pd.DataFrame({"response_GPT": [None, "ok"], "response_Claude": [None, None]})
    total, done = count_response_tasks(
        df,
        ["GPT", "Claude"],
        resume=False,
        response_is_done_fn=response_is_done,
    )
    assert total == 4
    assert done == 0


def test_task_progress_zero_total():
    bar = task_progress(0, "test")
    bar.update(1)
    bar.close()
