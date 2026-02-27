#ifndef RACER_SIMPLE_H
#define RACER_SIMPLE_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "raylib.h"

#define ACT_LEFT    0
#define ACT_FORWARD 1
#define ACT_RIGHT   2
#define ACT_BRAKE   3

#define PI2 (PI * 2)

#define MAX_TRACK_POINTS 512
#define MAX_WHISKERS 32
#define MAX_NPCS 16
#define NUM_LANES 3

typedef struct Client Client;

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    int lane;
    float track_pos;
    float px, py;
} NPC;

typedef struct RacerSimple {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;

    // Car
    float px, py, ang, v;
    float maxv, min_v;
    float turn_rate;
    float accel, decel;

    // Track
    Vector2 centerline[MAX_TRACK_POINTS];
    Vector2 inner_edge[MAX_TRACK_POINTS];
    Vector2 outer_edge[MAX_TRACK_POINTS];
    int total_points;
    int near_idx;

    // Whiskers
    int num_whiskers;
    float whisker_lengths[MAX_WHISKERS];
    Vector2 whisker_dirs[MAX_WHISKERS];
    float w_ang;
    float max_whisker_length;
    float car_radius;

    // NPCs
    int num_npcs;
    NPC npcs[MAX_NPCS];
    float lane_speeds[NUM_LANES];
    float npc_radius;

    // Lives
    int lives, max_lives;
    int invincibility;
    float collision_penalty;

    // Reward
    float prev_angle;
    float reward_scale;

    // Config
    int width, height;
    int track_width;
    int num_points;
    int bezier_resolution;
    int frameskip;
    int tick;
    float score;

    // Precomputed
    float inv_width, inv_height;
    float inv_bezier_res;

    unsigned int rng;
    int i;

    // Render
    int render;
    Client* client;
} RacerSimple;

struct Client {
    float width;
    float height;
};

static inline Vector2 vec2_normalize(Vector2 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y);
    if (len < 1e-5f) return (Vector2){0, 0};
    return (Vector2){v.x / len, v.y / len};
}

static inline Vector2 vec2_perp(Vector2 v) {
    return (Vector2){-v.y, v.x};
}

static inline int ray_seg_intersect(Vector2 origin, Vector2 dir, float max_len,
                                    Vector2 a, Vector2 b, float* t_out) {
    Vector2 seg = {b.x - a.x, b.y - a.y};
    Vector2 diff = {a.x - origin.x, a.y - origin.y};
    float cross_ds = dir.x * seg.y - dir.y * seg.x;
    if (fabsf(cross_ds) < 1e-3f) return 0;
    float t = (diff.x * seg.y - diff.y * seg.x) / cross_ds;
    float u = (diff.x * dir.y - diff.y * dir.x) / cross_ds;
    if (t >= 0.0f && t <= max_len && u >= 0.0f && u <= 1.0f) {
        *t_out = t;
        return 1;
    }
    return 0;
}

static inline int ray_circle_intersect(Vector2 origin, Vector2 dir, float max_len,
                                       Vector2 center, float radius, float* t_out) {
    float dx = origin.x - center.x;
    float dy = origin.y - center.y;
    float a = dir.x * dir.x + dir.y * dir.y;
    float b = 2.0f * (dx * dir.x + dy * dir.y);
    float c = dx * dx + dy * dy - radius * radius;
    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return 0;
    float sqrt_disc = sqrtf(disc);
    float t = (-b - sqrt_disc) / (2.0f * a);
    if (t < 0.0f) t = (-b + sqrt_disc) / (2.0f * a);
    if (t >= 0.0f && t <= max_len) {
        *t_out = t;
        return 1;
    }
    return 0;
}

static inline float point_seg_dist_sq(Vector2 p, Vector2 a, Vector2 b) {
    float abx = b.x - a.x, aby = b.y - a.y;
    float apx = p.x - a.x, apy = p.y - a.y;
    float t = (apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-12f);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    float cx = a.x + t * abx - p.x;
    float cy = a.y + t * aby - p.y;
    return cx * cx + cy * cy;
}

static inline Vector2 cubic_bezier(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t) {
    float u = 1.0f - t;
    float uu = u * u, uuu = uu * u;
    float tt = t * t, ttt = tt * t;
    return (Vector2){
        uuu*p0.x + 3*uu*t*p1.x + 3*u*tt*p2.x + ttt*p3.x,
        uuu*p0.y + 3*uu*t*p1.y + 3*u*tt*p2.y + ttt*p3.y
    };
}

void generate_track(RacerSimple* env);
void update_nearest(RacerSimple* env);
void compute_whiskers(RacerSimple* env);
void compute_reward(RacerSimple* env);
void compute_observations(RacerSimple* env);
void add_log(RacerSimple* env);
void spawn_npcs(RacerSimple* env);
void update_npcs(RacerSimple* env);
void check_npc_collision(RacerSimple* env);
void c_reset(RacerSimple* env);
void init(RacerSimple* env);
void allocate(RacerSimple* env);
void step_frame(RacerSimple* env, float action);
void c_step(RacerSimple* env);
Client* make_client(RacerSimple* env);
void close_client(Client* client);
void c_render(RacerSimple* env);
void c_close(RacerSimple* env);
void free_allocated(RacerSimple* env);

#endif
