import subprocess
from pathlib import Path


pool = None


def make_cmd(cmd):
    def clean(c):
        if isinstance(c, Path):
            return str(c.absolute())
        else:
            return c

    return [clean(c) for c in cmd]


def synchronous_run(cmd, logfile, kwargs):
    if logfile is not None:
        with open(logfile, "a+") as log:
            log.write(" ".join(cmd))
            log.write("\n")
            log.flush()
            return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    else:
        return subprocess.run(cmd, **kwargs)


def synchronous_check_output(cmd, kwargs):
    return subprocess.check_output(cmd, **kwargs)


def setup(procs):
    global pool


#    multiprocessing.set_start_method("forkserver")
#    pool = multiprocessing.Pool(processes=procs)


def run(cmd, logfile, **kwargs):
    cmd = make_cmd(cmd)
    global pool
    if pool is not None:
        return pool.apply(synchronous_run, (cmd, logfile, kwargs))
    else:
        return synchronous_run(cmd, logfile, kwargs)


def call(cmd, logfile, **kwargs):
    cmd = make_cmd(cmd)
    global pool
    if pool is not None:
        return run(cmd, logfile, **kwargs).returncode
    else:
        if logfile is not None:
            with open(logfile, "a+") as log:
                log.write(" ".join(cmd))
                log.write("\n")
                log.flush()
            return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        else:
            return subprocess.call(cmd, **kwargs)


def check_output(cmd, **kwargs):
    cmd = make_cmd(cmd)
    global pool
    if pool is not None:
        return pool.apply(synchronous_check_output, (cmd, kwargs))
    else:
        return synchronous_check_output(cmd, kwargs)


def shutdown():
    global pool
    if pool is not None:
        pool.terminate()
        pool.join()
        pool = None


def run_java(cmd, logfile, check=True, **kwargs):
    call_java(cmd, logfile, check=check, **kwargs)


def call_java(cmd, logfile, **kwargs):
    cmd = make_cmd(cmd)
    try:
        if "timeout" not in kwargs:
            kwargs["timeout"] = 3600
        # , "-XX:+HeapDumpOnOutOfMemoryError"
        run(["java", "-Xss1g", "-Xmx10g"] + cmd, logfile, **kwargs)
    finally:
        # logging.info("Stopped {}".format(cmd))
        if "cwd" in kwargs:
            clear_failed_java(kwargs["cwd"])
        else:
            clear_failed_java(Path("../evogfuzzcoded"))


def clear_failed_java(cwd):
    for dump in cwd.glob("*.dmp"):
        dump.unlink()
    for dump in cwd.glob("*.phd"):
        dump.unlink()
    for dump in cwd.glob("*.trc"):
        dump.unlink()
