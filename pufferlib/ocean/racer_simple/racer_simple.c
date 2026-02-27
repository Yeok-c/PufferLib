#include "racer_simple.h"

void generate_track(RacerSimple* env) {
    int n = env->num_points;
    float cx = env->width * 0.5f;
    float cy = env->height * 0.5f;
    float base_r = env->height * 0.5f;

    float freq1 = 2.0f + (rand() % 5);
    float amp1  = (1.0f / freq1) * (0.9f + 0.2f * (rand() % 100) / 100.0f);
    float ph1   = PI2 * (rand() % 100) / 100.0f;

    float freq2 = 1.0f + (rand() % 2);
    float amp2  = 0.2f + 0.2f * (rand() % 100) / 100.0f;
    float ph2   = PI2 * (rand() % 100) / 100.0f;

    float freq3 = 10.0f + 0.5f * (rand() % 3);
    float amp3  = 0.3f + 0.1f * (rand() % 100) / 100.0f;
    float ph3   = PI2 * (rand() % 100) / 100.0f;

    Vector2 controls[32];
    for (int i = 0; i < n; i++) {
        float a = (PI2 * i) / n;
        float rv = amp1*cosf(freq1*a+ph1) + amp2*cosf(freq2*a+ph2) + amp3*cosf(freq3*a+ph3);
        float r = base_r + base_r * 0.5f * rv;
        controls[i] = (Vector2){cx + r * 1.0f * cosf(a), cy + r * 0.6f * sinf(a)};
    }

    float tw2 = env->track_width * 0.5f;
    for (int i = 0; i < n; i++) {
        if (controls[i].x < tw2) controls[i].x = tw2;
        if (controls[i].x > env->width - tw2) controls[i].x = env->width - tw2;
        if (controls[i].y < tw2) controls[i].y = tw2;
        if (controls[i].y > env->height - tw2) controls[i].y = env->height - tw2;
    }

    int idx = 0;
    for (int i = 0; i < n; i++) {
        Vector2 p0 = controls[i];
        Vector2 p3 = controls[(i + 1) % n];
        Vector2 prev = controls[(i - 1 + n) % n];
        Vector2 next = controls[(i + 2) % n];

        Vector2 d1 = vec2_normalize((Vector2){p3.x - prev.x, p3.y - prev.y});
        Vector2 d2 = vec2_normalize((Vector2){next.x - p0.x, next.y - p0.y});
        float dist = sqrtf((p3.x-p0.x)*(p3.x-p0.x) + (p3.y-p0.y)*(p3.y-p0.y));
        float cl = dist * 0.35f;

        Vector2 p1 = {p0.x + d1.x*cl, p0.y + d1.y*cl};
        Vector2 p2 = {p3.x - d2.x*cl, p3.y - d2.y*cl};

        for (int j = 0; j < env->bezier_resolution && idx < MAX_TRACK_POINTS - 1; j++) {
            float t = (float)j * env->inv_bezier_res;
            env->centerline[idx++] = cubic_bezier(p0, p1, p2, p3, t);
        }
    }
    env->total_points = idx;

    float hw = env->track_width * 0.5f;
    for (int i = 0; i < idx; i++) {
        Vector2 cur = env->centerline[i];
        Vector2 nxt = env->centerline[(i + 1) % idx];
        Vector2 tangent = vec2_normalize((Vector2){nxt.x - cur.x, nxt.y - cur.y});
        Vector2 normal = vec2_perp(tangent);
        env->inner_edge[i] = (Vector2){cur.x - normal.x*hw, cur.y - normal.y*hw};
        env->outer_edge[i] = (Vector2){cur.x + normal.x*hw, cur.y + normal.y*hw};
    }
}

static void npc_world_pos(RacerSimple* env, NPC* npc) {
    int idx = (int)npc->track_pos;
    float frac = npc->track_pos - idx;
    idx = idx % env->total_points;
    if (idx < 0) idx += env->total_points;
    int nxt = (idx + 1) % env->total_points;
    float lane_t = (2.0f * npc->lane + 1.0f) / 6.0f;
    Vector2 in0 = env->inner_edge[idx], out0 = env->outer_edge[idx];
    Vector2 in1 = env->inner_edge[nxt], out1 = env->outer_edge[nxt];
    float px0 = in0.x + lane_t * (out0.x - in0.x);
    float py0 = in0.y + lane_t * (out0.y - in0.y);
    float px1 = in1.x + lane_t * (out1.x - in1.x);
    float py1 = in1.y + lane_t * (out1.y - in1.y);
    npc->px = px0 + frac * (px1 - px0);
    npc->py = py0 + frac * (py1 - py0);
}

void spawn_npcs(RacerSimple* env) {
    for (int l = 0; l < NUM_LANES; l++)
        env->lane_speeds[l] = env->maxv * (0.3f + 0.4f * (rand() % 100) / 100.0f) / 20.0f;

    int spacing = env->total_points / (env->num_npcs > 0 ? env->num_npcs : 1);
    for (int n = 0; n < env->num_npcs; n++) {
        env->npcs[n].lane = rand() % NUM_LANES;
        env->npcs[n].track_pos = (float)((n * spacing + rand() % (spacing > 1 ? spacing : 1)) % env->total_points);
        npc_world_pos(env, &env->npcs[n]);
    }
}

void update_npcs(RacerSimple* env) {
    for (int n = 0; n < env->num_npcs; n++) {
        NPC* npc = &env->npcs[n];
        npc->track_pos -= env->lane_speeds[npc->lane];
        if (npc->track_pos < 0)
            npc->track_pos += env->total_points;
        npc_world_pos(env, npc);
    }
}

void update_nearest(RacerSimple* env) {
    float best = 1e9f;
    int best_i = env->near_idx;
    for (int off = 0; off <= 3; off++) {
        int i = (env->near_idx + off) % env->total_points;
        float dx = env->px - env->centerline[i].x;
        float dy = env->py - env->centerline[i].y;
        float d = dx*dx + dy*dy;
        if (d < best) { best = d; best_i = i; }
    }
    env->near_idx = best_i;
}

void compute_whiskers(RacerSimple* env) {
    float max_len = env->max_whisker_length;
    float inv_max = 1.0f / max_len;
    int nw = env->num_whiskers;
    int window = 10;

    update_nearest(env);

    Vector2 car = {env->px, env->py};

    for (int w = 0; w < nw; w++) {
        Vector2 dir = env->whisker_dirs[w];
        float min_hit = max_len;

        for (int off = -window/2; off <= window/2; off++) {
            int i = (env->near_idx + off + env->total_points) % env->total_points;
            int ni = (i + 1) % env->total_points;
            float t;
            if (ray_seg_intersect(car, dir, max_len, env->inner_edge[i], env->inner_edge[ni], &t))
                if (t < min_hit) min_hit = t;
            if (ray_seg_intersect(car, dir, max_len, env->outer_edge[i], env->outer_edge[ni], &t))
                if (t < min_hit) min_hit = t;
        }

        for (int n = 0; n < env->num_npcs; n++) {
            Vector2 nc = {env->npcs[n].px, env->npcs[n].py};
            float t;
            if (ray_circle_intersect(car, dir, max_len, nc, env->npc_radius, &t))
                if (t < min_hit) min_hit = t;
        }

        env->whisker_lengths[w] = fminf(1.0f, fmaxf(0.0f, min_hit * inv_max));
    }

    float r_sq = env->car_radius * env->car_radius;
    for (int off = -window/2; off <= window/2; off++) {
        int i = (env->near_idx + off + env->total_points) % env->total_points;
        int ni = (i + 1) % env->total_points;
        if (point_seg_dist_sq(car, env->inner_edge[i], env->inner_edge[ni]) <= r_sq ||
            point_seg_dist_sq(car, env->outer_edge[i], env->outer_edge[ni]) <= r_sq) {
            for (int w = 0; w < nw; w++) env->whisker_lengths[w] = 0.0f;
            env->terminals[0] = 1;
            return;
        }
    }
}

void check_npc_collision(RacerSimple* env) {
    if (env->invincibility > 0) {
        env->invincibility--;
        return;
    }
    float threshold_sq = (env->car_radius + env->npc_radius) * (env->car_radius + env->npc_radius);
    for (int n = 0; n < env->num_npcs; n++) {
        float dx = env->px - env->npcs[n].px;
        float dy = env->py - env->npcs[n].py;
        if (dx*dx + dy*dy <= threshold_sq) {
            env->v = 0.0f;
            env->lives--;
            env->invincibility = 30;
            env->rewards[0] -= env->collision_penalty;
            env->score -= env->collision_penalty;
            if (env->lives <= 0) {
                env->terminals[0] = 1;
            }
            return;
        }
    }
}

void compute_reward(RacerSimple* env) {
    float cx = env->width * 0.5f;
    float cy = env->height * 0.5f;
    float angle = atan2f(env->py - cy, env->px - cx);
    if (angle < 0) angle += PI2;

    float delta = angle - env->prev_angle;
    if (delta > PI) delta -= PI2;
    if (delta < -PI) delta += PI2;

    if (delta > 0) {
        float speed_reward = (env->v / env->maxv) * env->reward_scale;
        env->rewards[0] += speed_reward;
        env->score += speed_reward;
    }
    env->prev_angle = angle;
}

void compute_observations(RacerSimple* env) {
    for (int w = 0; w < env->num_whiskers; w++)
        env->observations[w] = env->whisker_lengths[w];
    env->observations[env->num_whiskers] = env->v / env->maxv;
    env->observations[env->num_whiskers + 1] = (float)env->lives / (float)env->max_lives;
}

void add_log(RacerSimple* env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->score;
    env->log.score += env->score;
    env->log.n += 1;
}

static void update_whisker_dirs(RacerSimple* env) {
    int nw = env->num_whiskers;
    int full_circle = (env->w_ang >= PI - 0.01f);
    for (int w = 0; w < nw; w++) {
        float offset;
        if (full_circle) {
            offset = PI2 * (float)w / nw;
        } else {
            float frac = (nw > 1) ? (float)w / (nw - 1) : 0.5f;
            offset = -env->w_ang + 2.0f * env->w_ang * frac;
        }
        env->whisker_dirs[w] = (Vector2){cosf(env->ang + offset), sinf(env->ang + offset)};
    }
}

void c_reset(RacerSimple* env) {
    generate_track(env);
    spawn_npcs(env);

    int start = rand() % env->total_points;
    env->near_idx = start;
    env->px = env->centerline[start].x;
    env->py = env->centerline[start].y;

    int nxt = (start + 1) % env->total_points;
    float dx = env->centerline[nxt].x - env->px;
    float dy = env->centerline[nxt].y - env->py;
    env->ang = atan2f(dy, dx);

    update_whisker_dirs(env);

    env->v = env->maxv;
    env->lives = env->max_lives;
    env->invincibility = 0;
    for (int w = 0; w < env->num_whiskers; w++) env->whisker_lengths[w] = 0.5f;
    env->score = 0;
    env->tick = 0;

    float cx = env->width * 0.5f;
    float cy = env->height * 0.5f;
    env->prev_angle = atan2f(env->py - cy, env->px - cx);
    if (env->prev_angle < 0) env->prev_angle += PI2;

    compute_observations(env);
}

void init(RacerSimple* env) {
    env->inv_width = 1.0f / env->width;
    env->inv_height = 1.0f / env->height;
    env->inv_bezier_res = 1.0f / env->bezier_resolution;
    env->tick = 0;
    env->render = 0;
    env->client = NULL;

    srand(env->rng + env->i);
    generate_track(env);
}

void allocate(RacerSimple* env) {
    init(env);
    env->observations = (float*)calloc(env->num_whiskers + 2, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void step_frame(RacerSimple* env, float action) {
    int act = (int)action;
    if (act == ACT_LEFT) {
        env->ang += env->turn_rate;
    } else if (act == ACT_RIGHT) {
        env->ang -= env->turn_rate;
    } else if (act == ACT_FORWARD) {
        env->v += env->accel;
        if (env->v > env->maxv) env->v = env->maxv;
    } else if (act == ACT_BRAKE) {
        env->v -= env->decel;
        if (env->v < env->min_v) env->v = env->min_v;
    }
    if (env->ang > PI2) env->ang -= PI2;
    else if (env->ang < 0) env->ang += PI2;

    update_whisker_dirs(env);

    env->px += env->v * cosf(env->ang);
    env->py += env->v * sinf(env->ang);
    if (env->px < 0) env->px = 0;
    else if (env->px > env->width) env->px = env->width;
    if (env->py < 0) env->py = 0;
    else if (env->py > env->height) env->py = env->height;

    update_npcs(env);

    compute_whiskers(env);
    if (env->terminals[0]) {
        add_log(env);
        c_reset(env);
        return;
    }

    check_npc_collision(env);
    if (env->terminals[0]) {
        add_log(env);
        c_reset(env);
        return;
    }

    compute_reward(env);
}

void c_step(RacerSimple* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;

    float action = env->actions[0];
    for (int i = 0; i < env->frameskip; i++) {
        env->tick++;
        step_frame(env, action);
    }
    compute_observations(env);
}

Client* make_client(RacerSimple* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    InitWindow(env->width, env->height, "Racer Simple");
    SetTargetFPS(60 / env->frameskip);
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(RacerSimple* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        env->render = 1;
    }

    BeginDrawing();
    ClearBackground(DARKGREEN);

    for (int i = 0; i < env->total_points; i++) {
        int ni = (i + 1) % env->total_points;
        Vector2 a = env->inner_edge[i]; a.y = env->height - a.y;
        Vector2 b = env->inner_edge[ni]; b.y = env->height - b.y;
        DrawLineEx(a, b, 5.0f, BLACK);
        a = env->outer_edge[i]; a.y = env->height - a.y;
        b = env->outer_edge[ni]; b.y = env->height - b.y;
        DrawLineEx(a, b, 5.0f, BLACK);
    }

    for (int lane = 1; lane < NUM_LANES; lane++) {
        float lt = (float)lane / NUM_LANES;
        for (int i = 0; i < env->total_points; i += 3) {
            int ni = (i + 1) % env->total_points;
            Vector2 in0 = env->inner_edge[i], out0 = env->outer_edge[i];
            Vector2 in1 = env->inner_edge[ni], out1 = env->outer_edge[ni];
            Vector2 a = {in0.x + lt * (out0.x - in0.x), env->height - (in0.y + lt * (out0.y - in0.y))};
            Vector2 b = {in1.x + lt * (out1.x - in1.x), env->height - (in1.y + lt * (out1.y - in1.y))};
            DrawLineEx(a, b, 2.0f, (Color){180, 180, 180, 120});
        }
    }

    for (int n = 0; n < env->num_npcs; n++) {
        float nx = env->npcs[n].px;
        float ny = env->height - env->npcs[n].py;
        Color npc_col;
        switch (env->npcs[n].lane) {
            case 0: npc_col = (Color){200, 200, 200, 255}; break;
            case 1: npc_col = (Color){180, 100, 100, 255}; break;
            default: npc_col = (Color){100, 100, 180, 255}; break;
        }
        DrawCircle((int)nx, (int)ny, env->npc_radius, npc_col);
    }

    float cx = env->px;
    float cy = env->height - env->py;
    int visible = (env->invincibility <= 0) || ((env->invincibility / 3) % 2 == 0);
    if (visible) {
        DrawCircle((int)cx, (int)cy, env->car_radius, YELLOW);
    }

    float wlen = env->max_whisker_length;
    int nw = env->num_whiskers;
    for (int w = 0; w < nw; w++) {
        float len = env->whisker_lengths[w] * wlen;
        Vector2 tip = {cx + env->whisker_dirs[w].x * len,
                       cy - env->whisker_dirs[w].y * len};
        unsigned char r = (unsigned char)(255 * w / (nw > 1 ? nw - 1 : 1));
        unsigned char b = (unsigned char)(255 - r);
        Color col = {r, 0, b, 255};
        DrawLineEx((Vector2){cx, cy}, tip, 5.0f, col);
    }

    char hud[64];
    snprintf(hud, sizeof(hud), "Lives: %d | Score: %.1f", env->lives, env->score);
    int text_width = MeasureText(hud, 24);
    DrawText(hud, env->width - text_width - 12, 10, 24, WHITE);

    EndDrawing();
}

void c_close(RacerSimple* env) {
    if (env->client != NULL) {
        close_client(env->client);
        env->client = NULL;
    }
}

void free_allocated(RacerSimple* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

#ifdef RACER_SIMPLE_DEMO
#include <time.h>
#include <signal.h>
#include "puffernet.h"

static volatile int g_quit = 0;
static void handle_sigint(int sig) { (void)sig; g_quit = 1; }

void demo() {
    signal(SIGINT, handle_sigint);
    int num_whiskers = 10;
    int input_size = num_whiskers + 2;
    int hidden = 128;
    int num_actions = 4;
    int total_weights = (input_size * hidden + hidden)
                      + (4 * hidden * hidden + 4 * hidden)
                      + (hidden + 1)
                      + (hidden * num_actions + num_actions);

    Weights* weights = NULL;
    LinearLSTM* net = NULL;
    const char* weights_path = "resources/racer_simple/puffer_racer_simple_weights.bin";
    FILE* wf = fopen(weights_path, "rb");
    if (wf) {
        fseek(wf, 0, SEEK_END);
        long file_bytes = ftell(wf);
        fclose(wf);
        long expected_bytes = (long)total_weights * (long)sizeof(float);
        if (file_bytes == expected_bytes) {
            weights = load_weights(weights_path, total_weights);
            int logit_sizes[1] = {num_actions};
            net = make_linearlstm(weights, 1, input_size, logit_sizes, 1);
        } else {
            fprintf(stderr, "Warning: weights file wrong size (%ld bytes, expected %ld) -- using random actions\n",
                    file_bytes, expected_bytes);
        }
    } else {
        fprintf(stderr, "Warning: no weights file found -- using random actions\n");
        fprintf(stderr, "  Train with: puffer train racer_simple\n");
        fprintf(stderr, "  Export with: cd pufferlib/ocean/racer_simple && bash build.sh\n");
    }

    RacerSimple env = {
        .frameskip = 1,
        .width = 1080,
        .height = 720,
        .track_width = 75,
        .max_whisker_length = 100,
        .num_whiskers = num_whiskers,
        .w_ang = 0.784,
        .turn_rate = 0.0785,
        .maxv = 5,
        .min_v = 1.0,
        .accel = 0.2,
        .decel = 0.3,
        .reward_scale = 1.0,
        .car_radius = 10,
        .npc_radius = 10,
        .num_npcs = 5,
        .max_lives = 3,
        .collision_penalty = 0.2,
        .num_points = 16,
        .bezier_resolution = 4,
        .rng = 6,
        .i = 1,
    };

    allocate(&env);
    env.client = make_client(&env);
    signal(SIGINT, handle_sigint);
    c_reset(&env);

    int frame = 0;
    SetTargetFPS(60);
    while (!WindowShouldClose() && !g_quit) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = ACT_FORWARD;
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) env.actions[0] = ACT_LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = ACT_RIGHT;
            if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) env.actions[0] = ACT_BRAKE;
        } else if (frame % 4 == 0) {
            if (net) {
                int* actions = (int*)env.actions;
                forward_linearlstm(net, env.observations, actions);
                env.actions[0] = actions[0];
            } else {
                env.actions[0] = rand() % num_actions;
            }
        }

        frame = (frame + 1) % 4;
        c_step(&env);
        c_render(&env);
    }

    if (net) free_linearlstm(net);
    if (weights) free(weights);
    free_allocated(&env);
}

int main() {
    demo();
}
#endif
