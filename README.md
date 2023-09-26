# Rutentiny
A small, kinda-multithreaded Minecraft Classic server written in Python.

NOTE: This hasn't been tested that thoroughly. It's probably stable enough for
  a toy server, but there's definitely rough edges. Report any bugs or crashes
  to Ada <sarahsooup@protonmail.com>

A handful of CPE extensions are supported. Map saving and loading is
  implemented, along with a config system. The example config file shows a
  schema for each of the supported keys.

Currently, Rutentiny doesn't perform any HTTP heartbeat to either Mojang or
  Classicube, so servers running it won't appear on their server lists.

Chat logs are printed with a timestamp to stdout and some commands can be issued
  though stdin. The server can be stopped with `stop`, ^C or an EOF. Currently,
  the server console and operator chat commands aren't equally implemented yet.

If the opensimplex module is installed, it greatly improves the quality of some
  of the map generators.

The program code, example maps and example config file are freely available
  under the Creative Commons Zero license. See COPYING for more info.
