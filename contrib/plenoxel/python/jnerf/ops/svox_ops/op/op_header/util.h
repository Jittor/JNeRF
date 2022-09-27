#pragma once
#define _EXP(x) __expf(x) // FAST EXP
#define _SIGMOID(x) (1 / (1 + _EXP(-(x))))