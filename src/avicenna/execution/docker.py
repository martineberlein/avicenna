import pkgutil

import tempfile

import src.avicenna.execution.external_exec as execute
import logging
from pathlib import Path


PACKAGE_NAME = "avicenna.execution"


class Container:
    def __init__(self, basedir: Path, name: str):
        self.__basedir = basedir.resolve()
        self.__imagename = self.__basedir.name
        self.__name = name
        self.__running = False

    def start(self, username: str = "experimentator"):
        self.create_image()

        # and start the container
        logging.info(f"Starting container {self.__name}...")
        proc = execute.run(
            ["docker", "run", "-dt", "--name", self.__name, self.__imagename], None
        )
        proc.check_returncode()
        self.__running = True

        # get the script files in place
        with tempfile.TemporaryDirectory() as tmpdir:
            alhazendir = Path(tmpdir) / "alhazen"
            alhazendir.mkdir(parents=True)
            needed_files = ["helpers.py", "oracles.py", "external_exec.py"]
            for file in needed_files:
                with open(alhazendir / file, "wb") as sc:
                    data = pkgutil.get_data(PACKAGE_NAME, file)
                    assert data is not None
                    sc.write(data)
            # file path are within the docker container, and therefore, hardcoded and absolute
            self.copy_into(
                [alhazendir],
                self.container_root_dir(username) / "alhazen_scripts",
                username=username,
            )
        logging.info(
            "Container {} for {} is running".format(self.__name, self.__imagename)
        )

    def container_root_dir(self, username: str) -> Path:
        if "root" == username:
            return Path("/root/Desktop/")
        else:
            return Path(f"/home/{username}/")

    @property
    def name(self) -> str:
        return self.__name

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def create_image(self):
        # check whether the image exists
        output = execute.check_output(["docker", "images", "-q", self.__imagename])
        if 0 == len(output):
            logging.info(f"Creating image {self.__imagename}...")
            execute.run(
                ["docker", "build", "-t", self.__imagename, "."],
                None,
                cwd=str(self.__basedir.resolve()),
            )
        else:
            logging.info(f"Image {self.__imagename} has id {output}")

    def copy_into(self, files, targetdir, username: str = "experimentator"):
        if not self.__running:
            raise AssertionError("Machine is stopped already.")
        name = self.__name
        for file in files:
            execute.run(
                [
                    "docker",
                    "cp",
                    str(file.absolute()),
                    "{name}:{path}/".format(name=name, path=str(targetdir)),
                ],
                None,
                check=True,
            )
            execute.run(
                [
                    "docker",
                    "exec",
                    "-u",
                    "root",
                    name,
                    "chown",
                    "-R",
                    f"{username}:{username}",
                    str(targetdir),
                ],
                None,
                check=False,
            )

    def extract(self, files, targetdir, username: str = "experimentator"):
        if not self.__running:
            raise AssertionError("Machine is stopped already.")
        name = self.__name
        for file in files:
            cmd = [
                "docker",
                "cp",
                "{name}:{path}".format(name=name, path=str(file)),
                str(targetdir.absolute()),
            ]
            execute.run(cmd, None, check=True)

    def check_output(self, cmd, cwd=None):
        if not self.__running:
            raise AssertionError("Machine is stopped already.")
        full_cmd = ["docker", "exec"]
        if cwd is not None:
            full_cmd = full_cmd + ["-w", cwd]
        full_cmd = full_cmd + [self.__name] + cmd
        return execute.check_output(full_cmd)

    def stop(self):
        if self.__running:
            logging.info("Stopping container {}".format(self.__name))
            killout = execute.check_output(["docker", "kill", self.__name])
            logging.info(f"Kill: {killout}")
            rmout = execute.check_output(["docker", "rm", self.__name])
            logging.info(f"rm: {rmout}")
            self.__running = False
