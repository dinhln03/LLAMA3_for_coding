from conans import ConanFile, CMake

class LibB(ConanFile):
    name = "libB"
    version = "0.0"

    settings = "os", "arch", "compiler", "build_type"
    options = {"shared": [True, False]}
    default_options = {"shared": False}

    generators = "cmake"

    scm = {"type": "git",
           "url": "auto",
           "revision": "auto"}

    exports_sources = "LICENSE" # to avoid build info bug

    def requirements(self):
        self.requires("libA/[>=0.0]@demo/testing")
        self.requires("libF/0.0@demo/testing")

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        cmake.install()

    def package(self):
        self.copy("LICENSE", dst="licenses")

    def package_info(self):
        self.cpp_info.libs = ["libB",]
