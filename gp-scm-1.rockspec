package = "gp"
version = "scm-1"

source = {
   url = "git://github.com/j-wilson/gpTorch7.git",
}

description = {
   summary = "Torch7 Gaussian Processes Package",
   detailed = [[
   ]],
   homepage = "https://github.com/j-wilson/gpTorch7",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "optim"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}