#include "racer_simple.c"

#define Env RacerSimple
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->track_width = unpack(kwargs, "track_width");
    env->max_whisker_length = unpack(kwargs, "max_whisker_length");
    env->num_whiskers = unpack(kwargs, "num_whiskers");
    env->w_ang = unpack(kwargs, "w_ang");
    env->turn_rate = unpack(kwargs, "turn_rate");
    env->maxv = unpack(kwargs, "maxv");
    env->min_v = unpack(kwargs, "min_v");
    env->accel = unpack(kwargs, "accel");
    env->decel = unpack(kwargs, "decel");
    env->frameskip = unpack(kwargs, "frameskip");
    env->num_points = unpack(kwargs, "num_points");
    env->bezier_resolution = unpack(kwargs, "bezier_resolution");
    env->reward_scale = unpack(kwargs, "reward_scale");
    env->car_radius = unpack(kwargs, "car_radius");
    env->num_npcs = unpack(kwargs, "num_npcs");
    env->npc_radius = unpack(kwargs, "npc_radius");
    env->max_lives = unpack(kwargs, "max_lives");
    env->collision_penalty = unpack(kwargs, "collision_penalty");
    env->rng = unpack(kwargs, "rng");
    env->i = unpack(kwargs, "i");

    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
