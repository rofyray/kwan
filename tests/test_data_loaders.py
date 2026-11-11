"""Every dataset loader, on tiny synthetic files written here in each dataset's real layout.

No real row is ever committed: ids, times and answers below are made up. The files copy only the
column names and the quirks of the real downloads (encodings, padded sequences, decimal ids).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kwan.data import loaders, schema
from kwan.data.loaders import ammore, assistments, eedi, junyi, xes3g5m

CORE = list(schema.CORE)


def check_schema(df, key, commercial, subset=None):
    """Common checks: core columns first, valid, sorted, and the dataset facts are attached."""
    assert list(df.columns)[:5] == CORE
    assert all(c.startswith("meta_") for c in df.columns[5:])
    assert schema.validate(df) is df
    assert df.index.tolist() == list(range(len(df)))
    info = df.attrs["dataset"]
    assert info["key"] == key and info["commercial_params_ok"] is commercial and info["licence"]
    assert info.get("subset") == subset
    assert set(df.correct.unique()) <= {0, 1}


# ---------------------------------------------------------------- ASSISTments 2009

A09 = """order_id,user_id,problem_id,skill_id,correct,original,skill_name
50,2,10,1,1,1,Addition
30,1,11,2,0,1,Café sums
40,1,12,,1,1,
60,1,13,2,1,0,Scaffold step
10,2,14,3,1,1,First copy
10,2,14,7,0,1,Second skill tag of the same attempt
20,1,15,2,1,1,Subtraction
"""


def test_assistments_2009_wraps_the_v1_loader(tmp_path):
    path = tmp_path / "skill_builder_data_corrected.csv"
    path.write_bytes(A09.encode("latin-1"))
    df = assistments.load_2009(path)
    check_schema(df, "assistments_2009", True)
    assert list(df.columns) == CORE
    assert df.user_id.tolist() == [1, 1, 2, 2]
    assert df.t.tolist() == [20, 30, 10, 50]  # the order id is the time
    assert df.item_id.tolist() == [15, 11, 14, 10]
    assert df.skill_id.tolist() == [2, 2, 3, 1]
    assert df.correct.tolist() == [1, 0, 1, 1]

    from kwan.eval import kt

    v1 = kt.load(path)  # same rows in the same order as the v1 loader
    assert (v1.user_id.tolist(), v1.problem_id.tolist(), v1.correct.tolist()) == (
        df.user_id.tolist(),
        df.item_id.tolist(),
        df.correct.tolist(),
    )


# ---------------------------------------------------------------- ASSISTments 2012

A12 = (
    '"problem_log_id","skill","problem_id","user_id","start_time","end_time","correct","actions","skill_id"\n'
    '"901","Area","70","5","2012-09-28 15:11:27","2012-09-28 15:11:36.856","1","--- \n- - start\n  - 1348859487000","12"\n'
    '"902","","71","5","2012-09-28 15:12:00","2012-09-28 15:12:30","0","--- \n- - start","" \n'
    '"903","Area","72","4","2012-10-01 09:00:00","2012-10-01 09:00:20","0.5","x","12"\n'
    '"904","Mean","73","4","2012-10-01 09:00:00","2012-10-01 09:00:40","0","x","13"\n'
    '"900","Mean","74","4","2012-10-01 09:00:00.000","2012-10-01 09:01:40","1","x","13"\n'
    '"905","Area","70","5","2012-09-27 08:00:00","2012-09-27 08:00:09","0","Café","12"\n'
)


def test_assistments_2012_reads_in_chunks_and_filters(tmp_path):
    folder = tmp_path / "assist2012"
    folder.mkdir()
    (folder / assistments.FILE_2012).write_bytes(A12.replace('"" \n', '""\n').encode("latin-1"))
    df = assistments.load_2012(folder, chunk_rows=2)  # three chunks, quoted fields with line breaks
    check_schema(df, "assistments_2012", True)
    assert list(df.columns) == [*CORE, "meta_problem_log_id"]
    # dropped: 902 (no skill), 903 (partial credit). Order: student, start time, then problem log id.
    assert df.meta_problem_log_id.tolist() == [900, 904, 905, 901]
    assert df.user_id.tolist() == [4, 4, 5, 5]
    assert df.item_id.tolist() == [74, 73, 70, 70]
    assert df.skill_id.tolist() == [13, 13, 12, 12]
    assert df.correct.tolist() == [1, 0, 0, 1]
    assert pd.api.types.is_datetime64_any_dtype(df.t)
    assert str(df.user_id.dtype) == "int32" and str(df.correct.dtype) == "int8"
    pd.testing.assert_frame_equal(df, assistments.load_2012(folder / assistments.FILE_2012))  # file or folder


# ---------------------------------------------------------------- ASSISTments 2015


def test_assistments_2015_uses_the_skill_builder_as_item_and_skill(tmp_path):
    path = tmp_path / assistments.FILE_2015
    path.write_text(
        "user_id,log_id,sequence_id,correct\n9,300,7014,0\n9,100,7014,1\n8,200,7012,0.25\n8,250,7012,1\n9,200,5898,1\n",
        encoding="utf-8",
    )
    df = assistments.load_2015(tmp_path)
    check_schema(df, "assistments_2015", True)
    assert df.attrs["dataset"]["items_are_skills"] is True
    assert df.user_id.tolist() == [8, 9, 9, 9]  # the partial-credit row is dropped
    assert df.t.tolist() == [250, 100, 200, 300]
    assert df.item_id.tolist() == df.skill_id.tolist() == [7012, 7014, 5898, 7014]
    assert df.correct.tolist() == [1, 1, 1, 0]


# ---------------------------------------------------------------- ASSISTments 2017


def test_assistments_2017_numbers_the_skill_names(tmp_path):
    path = tmp_path / assistments.FILE_2017
    path.write_text(
        "studentId,MiddleSchoolId,action_num,skill,problemId,startTime,endTime,correct,original\n"
        "8,2,1,square-root,501,1096470350,1096470354,1,1\n"
        "8,2,2,area,502,1096470301,1096470350,0,1\n"
        "8,2,3,area,503,1096470350,1096470360,1,0\n"
        "3,2,4,,504,1096470400,1096470410,1,1\n"
        "3,2,5,mean,505,1096470500,1096470510,0,1\n"
        "3,2,6,area,502,1096470450,1096470460,1,1\n",
        encoding="utf-8",
    )
    df = assistments.load_2017(path)
    check_schema(df, "assistments_2017", True)
    assert df.attrs["skill_names"] == {0: "area", 1: "mean", 2: "square-root"}
    assert df.user_id.tolist() == [3, 3, 8, 8, 8]  # the row without a skill is dropped
    assert df.item_id.tolist() == [502, 505, 502, 501, 503]  # equal start times keep file order
    assert df.skill_id.tolist() == [0, 1, 0, 2, 0]
    assert df.correct.tolist() == [1, 0, 0, 1, 1]
    assert df.t.iloc[2] == pd.Timestamp("2004-09-29 15:05:01")
    assert df.meta_row.tolist() == [5, 4, 1, 0, 2]


# ---------------------------------------------------------------- Eedi


def eedi_dir(root, task):
    (root / "train_data").mkdir(parents=True)
    (root / "metadata").mkdir()
    (root / "train_data" / f"train_task_{task}.csv").write_text(
        "QuestionId,UserId,AnswerId,IsCorrect,CorrectAnswer,AnswerValue\n"
        "0,7,100,1,2,2\n"
        "1,7,101,0,3,1\n"
        "2,5,102,1,1,1\n"
        "0,5,103,0,2,4\n"
        "1,5,104,1,3,3\n"
        "2,7,105,1,1,1\n"
        "0,9,106,1,2,2\n",
        encoding="utf-8",
    )
    decimal = ".0" if task == "1_2" else ""  # the task 1/2 file writes the id as a decimal
    rows = [
        (100, "2020-01-06 18:53:00.000"),
        (101, "2019-10-30 14:34:00.000"),
        (102, "2020-02-01 09:00:00.000"),
        (103, "2020-02-01 09:00:00.000"),
        (104, "2020-01-01 09:00:00.000"),
        (105, "2020-03-01 09:00:00.000"),
    ]  # answer 106 has no metadata
    text = "AnswerId,DateAnswered,Confidence,GroupId,QuizId,SchemeOfWorkId\n"
    text += "".join(f"{a}{decimal},{d},,1,1,52562.0\n" for a, d in rows)
    if task == "1_2":
        text += ",2020-03-02 09:00:00.000,,1,1,\n"  # the real file has a few rows with no id
    (root / "metadata" / f"answer_metadata_task_{task}.csv").write_text(text, encoding="utf-8")
    (root / "metadata" / f"question_metadata_task_{task}.csv").write_text(
        'QuestionId,SubjectId\n0,"[3, 32, 49, 33]"\n1,"[3, 32]"\n2,"[3, 999, 49]"\n', encoding="utf-8"
    )
    (root / "metadata" / "subject_metadata.csv").write_bytes(
        "﻿SubjectId,Name,ParentId,Level\r\n3,Maths,NULL,0\r\n32,Number,3,1\r\n49,BIDMAS,32,2\r\n33,Basic Arithmetic,49,3\r\n".encode()
    )
    return root


def check_eedi(df, subset):
    check_schema(df, "eedi_2020", True, subset)
    assert list(df.columns) == [*CORE, "meta_answer_id"]
    assert df.user_id.tolist() == [5, 5, 5, 7, 7, 7]  # answer 106 (no time) is dropped
    assert df.meta_answer_id.tolist() == [104, 102, 103, 101, 100, 105]  # student, time, then answer id
    assert df.correct.tolist() == [1, 1, 0, 0, 1, 1]
    assert dict(zip(df.item_id, df.skill_id)) == {0: 33, 1: 32, 2: 49}  # deepest subject; unknown 999 never wins
    assert pd.api.types.is_datetime64_any_dtype(df.t)
    assert df.t.iloc[0] == pd.Timestamp("2020-01-01 09:00:00")


def test_eedi_task34_wraps_the_v1_loader(tmp_path):
    check_eedi(eedi.load_task34(eedi_dir(tmp_path / "data", "3_4")), "task34")


def test_eedi_task12_chunked_pyarrow_read_with_32_bit_ids(tmp_path):
    root = eedi_dir(tmp_path / "data", "1_2")
    df = eedi.load_task12(root, block_bytes=64)  # 64-byte blocks: every file is read in many chunks
    check_eedi(df, "task12")
    assert [str(df[c].dtype) for c in ("user_id", "item_id", "skill_id", "meta_answer_id")] == ["int32"] * 4
    assert str(df.correct.dtype) == "int8"
    pd.testing.assert_frame_equal(df, eedi.load_task12(root))  # the chunk size does not change the result


def test_eedi_task12_and_task34_agree_on_the_same_answers(tmp_path):
    a = eedi.load_task12(eedi_dir(tmp_path / "a", "1_2"))
    b = eedi.load_task34(eedi_dir(tmp_path / "b", "3_4"))
    for col in (*CORE, "meta_answer_id"):
        assert a[col].tolist() == b[col].tolist(), col


def test_eedi_task12_says_which_file_is_missing(tmp_path):
    root = eedi_dir(tmp_path / "data", "1_2")
    (root / "metadata" / "answer_metadata_task_1_2.csv").unlink()
    with pytest.raises(FileNotFoundError, match="answer_metadata_task_1_2.csv is missing"):
        eedi.load_task12(root)


# ---------------------------------------------------------------- AMMORE


def ammore_dir(root):
    root.mkdir(parents=True)
    (root / "AMMORE_student_answers.csv").write_text(
        "user_id,message_inserted_at,question,question_micro_lesson,student_response,human_label\n"
        "u2,01/05/24 10:00,What is 2 + 2?,G6.N1.1.1.1,4,correct_answer\n"
        "u1,01/03/24 09:30,What is 3 x 3?,G6.N2.2.3.10,6,wrong_answer\n"
        "u1,01/03/24 09:30,What is 2 + 2?,G6.N1.1.1.1,4,correct_answer\n"
        "u1,01/02/24 08:00,What is 3 x 3?,G6.N2.2.3.10,sticker,other\n"
        "u3,02/01/24 12:00,What is 9 - 4?,G6.N1.1.1.1,5,correct_answer\n"
        "u2,01/04/24 10:00,What is 3 x 3?,G6.N2.2.3.10,9,correct_answer\n",
        encoding="utf-8",
    )
    (root / "AMMORE_demographic_data.csv").write_text(
        "user ID,country,grade\nu1,Ghana,7\nu2,Kenya,6\nu3,Ghana,8\n", encoding="utf-8"
    )
    return root


def test_ammore_all_and_ghana(tmp_path):
    root = ammore_dir(tmp_path / "ammore")
    df = ammore.load(root)
    check_schema(df, "ammore_2024", False, "all")
    assert list(df.columns) == [*CORE, "meta_micro_lesson", "meta_country"] or set(df.columns) == {
        *CORE,
        "meta_micro_lesson",
        "meta_country",
    }
    assert len(df) == 5  # the "other" message is not an answer
    assert df.user_id.tolist() == ["u1", "u1", "u2", "u2", "u3"]
    assert df.correct.tolist() == [0, 1, 1, 1, 1]  # same-minute answers keep file order
    assert df.meta_country.tolist() == ["Ghana", "Ghana", "Kenya", "Kenya", "Ghana"]
    assert df.skill_id.nunique() == 2 and df.item_id.nunique() == 3
    assert dict(zip(df.meta_micro_lesson, df.skill_id)) == {"G6.N1.1.1.1": 0, "G6.N2.2.3.10": 1}
    assert pd.api.types.is_datetime64_any_dtype(df.t) and df.t.iloc[0] == pd.Timestamp("2024-01-03 09:30")

    gh = ammore.load(root, "ghana")
    check_schema(gh, "ammore_2024", False, "ghana")
    assert gh.user_id.tolist() == ["u1", "u1", "u3"] and set(gh.meta_country) == {"Ghana"}
    pd.testing.assert_frame_equal(gh, ammore.load_ghana(root))
    with pytest.raises(KeyError, match="unknown AMMORE subset"):
        ammore.load(root, "nigeria")


# ---------------------------------------------------------------- XES3G5M


def xes_dir(root):
    q = root / "XES3G5M" / "question_level"
    q.mkdir(parents=True)
    (q / "train_valid_sequences_quelevel.csv").write_text(
        "fold,uid,questions,concepts,responses,timestamps,selectmasks\n"
        '0,11,"5,6,7,-1","166_170,12,12,-1","1,0,1,-1","3000,1000,2000,-1","1,1,1,-1"\n'
        '0,11,"8,-1,-1,-1","40,-1,-1,-1","0,-1,-1,-1","500,-1,-1,-1","1,-1,-1,-1"\n'
        '3,10,"5,5,-1,-1","166_170,166_170,-1,-1","0,1,-1,-1","100,100,-1,-1","1,1,-1,-1"\n',
        encoding="utf-8",
    )
    (q / "test_quelevel.csv").write_text(
        'fold,uid,questions,concepts,responses,timestamps\n-1,12,"9,6","7_8_9,12","1,1","50,60"\n', encoding="utf-8"
    )
    return root


def test_xes3g5m_unpacks_padded_sequences(tmp_path):
    root = xes_dir(tmp_path / "xes3g5m")
    df = xes3g5m.load(root)
    check_schema(df, "xes3g5m", True)
    assert list(df.columns) == [*CORE, "meta_fold"]
    assert len(df) == 8  # the -1 padding is gone
    assert df.user_id.tolist() == [10, 10, 11, 11, 11, 11, 12, 12]
    assert df.item_id.tolist() == [5, 5, 8, 6, 7, 5, 9, 6]  # the two pieces of pupil 11 are joined by time
    assert df.skill_id.tolist() == [166, 166, 40, 12, 12, 166, 7, 12]  # first concept of "166_170"
    assert df.correct.tolist() == [0, 1, 0, 0, 1, 1, 1, 1]
    assert df.meta_fold.tolist() == [3, 3, 0, 0, 0, 0, -1, -1]
    assert df.t.iloc[2] == pd.Timestamp("1970-01-01 00:00:00.500")
    pd.testing.assert_frame_equal(df, xes3g5m.load(root / "XES3G5M"))  # the unpacked folder itself also works
    with pytest.raises(FileNotFoundError, match="question_level"):
        xes3g5m.load(tmp_path / "nowhere")


# ---------------------------------------------------------------- Junyi

JUNYI_HEADER = (
    "user_id,exercise,problem_type,problem_number,topic_mode,suggested,review_mode,time_done,time_taken,"
    "time_taken_attempts,correct,count_attempts,hint_used,count_hints,hint_time_taken_list,earned_proficiency,points_earned\n"
)


def junyi_dir(root):
    root.mkdir(parents=True)
    rows = [
        (30, "telling_time", "analog_word", 1, 1420714810324490, "false"),
        (20, "multiplication_1", "0", 6, 1403098400836660, "true"),
        (30, "telling_time", "digital", 2, 1420714700000000, "true"),
        (20, "multiplication_1", "0", 7, 1403098500000000, "true"),
        (20, "telling_time", "analog_word", 1, 1403098300000000, "false"),
        (30, "area_1", "", 1, 1420714900000000, "true"),
    ]
    body = "".join(
        f"{u},{ex},{pt},{n},false,false,false,{t},4,3&1,{c},2,false,0,,false,0\n" for u, ex, pt, n, t, c in rows
    )
    (root / junyi.LOG_FILE).write_text(JUNYI_HEADER + body, encoding="utf-8")
    return root


def test_junyi_problem_log(tmp_path):
    root = junyi_dir(tmp_path / "junyi")
    df = junyi.load(root, block_bytes=300)  # tiny blocks: ids must stay the same across chunks
    check_schema(df, "junyi_2015", False)
    assert list(df.columns) == CORE
    assert df.user_id.tolist() == [20, 20, 20, 30, 30, 30]
    names = df.attrs["skill_names"]
    assert set(names.values()) == {"telling_time", "multiplication_1", "area_1"}
    assert [names[s] for s in df.skill_id] == [
        "telling_time",
        "multiplication_1",
        "multiplication_1",
        "telling_time",
        "telling_time",
        "area_1",
    ]
    assert df.correct.tolist() == [0, 1, 1, 1, 0, 1]
    # item = exercise and problem template: telling_time has two templates, multiplication_1 one
    assert df.item_id.nunique() == 4
    assert df.item_id.iloc[1] == df.item_id.iloc[2] and df.item_id.iloc[3] != df.item_id.iloc[4]
    assert df.item_id.iloc[0] == df.item_id.iloc[4]
    assert df.t.iloc[0] == pd.Timestamp("2014-06-18 13:31:40")
    assert str(df.user_id.dtype) == "int32" and str(df.correct.dtype) == "int8"
    whole = junyi.load(root)
    for col in CORE:
        assert whole[col].tolist() == df[col].tolist()
    with pytest.raises(FileNotFoundError, match="junyi_ProblemLog_original.csv is missing"):
        junyi.load(tmp_path / "empty")


# ---------------------------------------------------------------- the loader table


def test_loader_table_covers_every_results_key(tmp_path):
    assert set(loaders.LOADERS) == {
        "assistments_2009",
        "assistments_2012",
        "assistments_2015",
        "assistments_2017",
        "eedi_2020_task34",
        "eedi_2020_task12",
        "ammore_2024",
        "ammore_2024:all",
        "ammore_2024:ghana",
        "xes3g5m",
        "junyi_2015",
    }
    for name in loaders.LOADERS:
        assert callable(loaders.loader(name))
    assert loaders.raw_path("ammore_2024:ghana", "raw") == Path("raw/ammore")
    with pytest.raises(KeyError, match="no loader for 'nope'"):
        loaders.loader("nope")

    gh = loaders.load("ammore_2024:ghana", ammore_dir(tmp_path / "raw" / "ammore"))
    assert len(gh) == 3
    assert len(loaders.load("ammore_2024", raw_dir=tmp_path / "raw")) == 5  # default place under the raw directory
    assert len(loaders.load("xes3g5m", xes_dir(tmp_path / "x"))) == 8


def test_loaded_frames_feed_the_models_directly(tmp_path):
    """A schema frame goes straight into the v1 models: `item_id` is read where v1 used `problem_id`."""
    from kwan.eval.kt import Elo2

    df = eedi.load_task12(eedi_dir(tmp_path / "data", "1_2"))
    a = Elo2().run(df, learn=True)
    b = Elo2().run(schema.with_problem_id(df).drop(columns="item_id"), learn=True)
    assert np.array_equal(a, b) and len(a) == len(df)
