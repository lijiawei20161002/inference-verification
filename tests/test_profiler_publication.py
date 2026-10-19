"""Keep credentials out of captures and preserve measurement records when publishing."""
import sqlite3
import pytest

from experiments.profile_vllm_serving import nsys_command, profiling_environment
from scripts.prepare_profiler_publication import sanitize_database


def test_nsys_command_requests_environment_discard():
    command = nsys_command("nsys", "capture", ["python", "server.py"])
    assert "--discard-environment=true" in command


def test_capture_environment_excludes_credentials_and_unknown_ambient_variables():
    ambient = dict(PATH="/bin", HOME="/home/test", LD_LIBRARY_PATH="/cuda/lib",
                   RUNPOD_API_KEY="private-value", JUPYTER_PASSWORD="private-password",
                   UNEXPECTED_CREDENTIAL="unrecognized-secret-name", OMP_NUM_THREADS="99")
    result = profiling_environment({"OMP_NUM_THREADS":"1","HF_HUB_OFFLINE":"1"}, ambient)
    assert result == dict(PATH="/bin",HOME="/home/test",LD_LIBRARY_PATH="/cuda/lib",
                          OMP_NUM_THREADS="1",HF_HUB_OFFLINE="1")


@pytest.mark.parametrize("credential_in_launch_environment", [True, False])
def test_redaction_removes_credentials_from_live_and_freed_sqlite_pages(tmp_path, monkeypatch, credential_in_launch_environment):
    secret = "publication-test-secret-never-a-real-key"
    if credential_in_launch_environment:
        monkeypatch.setenv("TEST_API_KEY", secret)
    else:
        monkeypatch.delenv("TEST_API_KEY", raising=False)
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript('''
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY,value TEXT);
            CREATE TABLE TARGET_INFO_SYSTEM_ENV(name TEXT,value TEXT);
            CREATE TABLE META_DATA_CAPTURE(name TEXT,value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INT,end INT,shortName INT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INT,end INT,nameId INT);
            CREATE TABLE NVTX_EVENTS(start INT,end INT,textId INT);
            INSERT INTO StringIds VALUES(1,'kernel_and_iteration');
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(10,20,1);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(1,3,1);
            INSERT INTO NVTX_EVENTS VALUES(0,5,1);
        ''')
        conn.execute("INSERT INTO StringIds VALUES(2,?)", ("TEST_API_KEY="+secret,))
        conn.execute("INSERT INTO TARGET_INFO_SYSTEM_ENV VALUES('TEST_API_KEY',?)", (secret,))
        conn.execute("INSERT INTO META_DATA_CAPTURE VALUES('PROCESS_0:ENVIRONMENT_VARIABLE',?)",
                     ("TEST_API_KEY="+secret,))
    assert secret.encode() in path.read_bytes()
    result = sanitize_database(path)
    assert secret.encode() not in path.read_bytes()
    assert result["original_sha256"] != result["published_sha256"]
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall() == [(10,20,1)]
        assert conn.execute("SELECT value FROM StringIds WHERE id=1").fetchone() == ('kernel_and_iteration',)
        assert not conn.execute("SELECT * FROM TARGET_INFO_SYSTEM_ENV").fetchall()
        assert not conn.execute("SELECT * FROM META_DATA_CAPTURE").fetchall()
