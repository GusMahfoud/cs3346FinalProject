import numpy as np
from poke_env.battle.battle import Battle
from poke_env.battle.side_condition import SideCondition
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

# -----------------------------
# TYPE DATA (GEN 9)
# -----------------------------
_GEN9_DATA = GenData.from_gen(9)
_TYPE_CHART = _GEN9_DATA.type_chart


class RewardCalculator:
    def __init__(
        self,
        enable_intermediate: bool = True,
        # terminal outcomes
        terminal_win: float = 100.0,
        terminal_loss: float = -100.0,
        terminal_draw: float = 0.0,
        # HP / damage shaping
        hp_weight: float = 1.5,          # weight on change in HP advantage
        ko_reward: float = 30.0,         # YOU KO THEM
        ko_penalty: float = -25.0,       # THEY KO YOU
        status_reward: float = 2.0,      # they gain status
        status_penalty: float = -1.5,    # you gain status
        # "Effectiveness" via damage chunk size
        type_effectiveness_reward: float = 6.0,    # big damage chunk
        type_effectiveness_penalty: float = -3.0,  # tiny damage
        # switching / tempo
        switch_penalty: float = -0.7,    # cost for switching
        turn_penalty: float = -0.08,     # tiny negative each turn (anti-stall)
        momentum_reward: float = 1.0,    # bonus after several favorable turns
        momentum_turns: int = 3,         # how many favorable turns before momentum bonus
    ):
        self.enable_intermediate = enable_intermediate
        self.terminal_win = terminal_win
        self.terminal_loss = terminal_loss
        self.terminal_draw = terminal_draw

        # Intermediate reward weights
        self.hp_weight = hp_weight
        self.ko_reward = ko_reward
        self.ko_penalty = ko_penalty
        self.status_reward = status_reward
        self.status_penalty = status_penalty
        self.type_effectiveness_reward = type_effectiveness_reward
        self.type_effectiveness_penalty = type_effectiveness_penalty
        self.switch_penalty = switch_penalty
        self.turn_penalty = turn_penalty
        self.momentum_reward = momentum_reward
        self.momentum_turns = momentum_turns

        # hazard / screen shaping constants (not exposed as args to keep API clean)
        self.hazard_sr_set_reward = 4.0
        self.hazard_spikes_layer_reward = 2.0
        self.hazard_tspikes_layer_reward = 2.5
        self.hazard_remove_reward = 3.0
        self.hazard_removed_penalty = -3.0
        self.hazard_switch_penalty_scale = 0.7  # extra cost when switching into hazards

        self.screen_set_reward = 3.0
        self.screen_lost_penalty = -2.0

        self.forced_switch_reward = 2.0
        self.forced_switch_hazard_bonus = 1.0

        # Track previous state (internal, per battle)
        self.prev_hp_advantage = 0.0
        self.prev_our_hp_pct = None
        self.prev_opp_hp_pct = None
        self.prev_our_fainted = 0
        self.prev_opp_fainted = 0
        self.prev_our_status = 0
        self.prev_opp_status = 0

        # Boost tracking (our active + opponent active)
        self.prev_our_boosts = {"atk": 0, "spa": 0, "spe": 0, "def": 0, "spd": 0}
        self.prev_opp_boosts = {"atk": 0, "spa": 0, "spe": 0, "def": 0, "spd": 0}

        # Hazard and screen tracking
        self.prev_hazards_our = {"sr": 0, "spikes": 0, "tspikes": 0, "webs": 0}
        self.prev_hazards_opp = {"sr": 0, "spikes": 0, "tspikes": 0, "webs": 0}
        self.prev_screens_our = {"reflect": 0, "lscreen": 0, "veil": 0}
        self.prev_screens_opp = {"reflect": 0, "lscreen": 0, "veil": 0}

        # Weather / terrain
        self.prev_weather = None
        self.prev_terrain = None

        # For forced-switch heuristics
        self.prev_opp_active_id = None

        # Momentum
        self.consecutive_favorable_turns = 0

    # =============================
    # HP HELPERS
    # =============================

    def _get_hp_percentage(self, battle: Battle, is_opponent: bool = False) -> float:
        """Total HP percentage for our side or opponent in the current Battle."""
        team = battle.opponent_team if is_opponent else battle.team
        total_hp = 0.0
        total_max_hp = 0.0

        for poke in team.values():
            if poke.max_hp:
                total_max_hp += poke.max_hp
                if poke.current_hp is not None:
                    total_hp += poke.current_hp

        if total_max_hp == 0:
            return 0.0
        return total_hp / total_max_hp

    def _count_fainted(self, battle: Battle, is_opponent: bool = False) -> int:
        """Count fainted Pokémon on one side."""
        team = battle.opponent_team if is_opponent else battle.team
        return sum(1 for poke in team.values() if poke.fainted)

    def _count_status_conditions(self, battle: Battle, is_opponent: bool = False) -> int:
        """Count Pokémon with status conditions on one side."""
        team = battle.opponent_team if is_opponent else battle.team
        return sum(1 for poke in team.values() if poke.status)

    # =============================
    # HAZARDS & SCREENS HELPERS
    # =============================

    def _extract_hazards(self, side_conditions) -> dict:
        """
        Extract hazard info from a side_conditions dict.
        Returns a dict: { sr, spikes, tspikes, webs }
        """
        sr = 1 if SideCondition.STEALTH_ROCK in side_conditions else 0
        spikes = side_conditions.get(SideCondition.SPIKES, 0)
        tspikes = side_conditions.get(SideCondition.TOXIC_SPIKES, 0)
        webs = 1 if SideCondition.STICKY_WEB in side_conditions else 0

        return {"sr": sr, "spikes": spikes, "tspikes": tspikes, "webs": webs}

    def _extract_screens(self, side_conditions) -> dict:
        """
        Extract screen-like conditions (Reflect, Light Screen, Aurora Veil).
        """
        reflect = 1 if SideCondition.REFLECT in side_conditions else 0
        lscreen = 1 if SideCondition.LIGHT_SCREEN in side_conditions else 0
        veil = 1 if SideCondition.AURORA_VEIL in side_conditions else 0

        return {"reflect": reflect, "lscreen": lscreen, "veil": veil}

    # =============================
    # TYPE EFFECTIVENESS
    # =============================

    def _type_effectiveness(self, move, target) -> float:
        """
        Approximate type effectiveness multiplier for move used on target.
        0, 0.25, 0.5, 1, 2, 4 based on the Gen 9 type chart.
        """
        if move is None or target is None:
            return 1.0

        try:
            atk_type = move.type
            if atk_type is None:
                return 1.0
            atk_id = to_id_str(atk_type)
        except Exception:
            return 1.0

        # Collect defending types
        def_types = []
        try:
            if target.type_1:
                def_types.append(to_id_str(target.type_1))
            if target.type_2 and target.type_2 != target.type_1:
                def_types.append(to_id_str(target.type_2))
        except Exception:
            # If we fail to read types, assume neutral
            return 1.0

        mult = 1.0
        chart_for_atk = _TYPE_CHART.get(atk_id, {})

        for d in def_types:
            mult *= chart_for_atk.get(d, 1.0)

        return mult

    # =============================
    # WEATHER / TERRAIN SHAPING
    # =============================

    def _small_weather_terrain_reward(self, battle: Battle) -> float:
        """
        Very small shaping for beneficial weather / terrain for our active Pokémon.
        Kept tiny so it doesn't dominate.
        """
        reward = 0.0
        active = battle.active_pokemon
        if active is None:
            return 0.0

        # --- Weather ---
        weather = battle.weather
        try:
            w_id = to_id_str(weather[0]) if isinstance(weather, tuple) else to_id_str(weather)
        except Exception:
            w_id = None

        if w_id is not None:
            types = {active.type_1, active.type_2}
            # simple heuristics
            if w_id == "raindance":
                if "water" in types:
                    reward += 0.2
                if "fire" in types:
                    reward -= 0.2
            elif w_id == "sunnyday":
                if "fire" in types:
                    reward += 0.2
                if "water" in types:
                    reward -= 0.2
            elif w_id == "sandstorm":
                # sand hurts non-rock/ground/steel most of the time
                if not (("rock" in types) or ("ground" in types) or ("steel" in types)):
                    reward -= 0.2
                else:
                    reward += 0.1
            elif w_id == "hail" or w_id == "snow":
                if "ice" in types:
                    reward += 0.2

        # --- Terrain ---
        fields = getattr(battle, "fields", {})
        terrain_id = None
        for f in fields:
            try:
                terrain_id = to_id_str(f)
                break
            except Exception:
                continue

        if terrain_id is not None:
            types = {active.type_1, active.type_2}
            if terrain_id == "electricterrain" and "electric" in types:
                reward += 0.2
            elif terrain_id == "grassyterrain" and "grass" in types:
                reward += 0.2
            elif terrain_id == "psychicterrain" and "psychic" in types:
                reward += 0.2
            elif terrain_id == "mistyterrain" and "dragon" not in types:
                # mild reward for being protected from status / dragon
                reward += 0.1

        return reward

    # =============================
    # MAIN TURN REWARD
    # =============================

    def compute_turn_reward(
        self,
        prev_battle,                # kept for API compatibility, not relied on
        current_battle: Battle,
        action_was_switch: bool = False,
    ) -> float:
        """
        Compute intermediate reward for a turn transition.

        Internally, this uses *its own* stored previous state rather than prev_battle,
        so it works even when prev_battle is just a dict snapshot.
        """
        if not self.enable_intermediate:
            return 0.0

        reward = 0.0

        # ---------- Current HP % ----------
        our_hp = self._get_hp_percentage(current_battle, is_opponent=False)
        opp_hp = self._get_hp_percentage(current_battle, is_opponent=True)
        hp_advantage = our_hp - opp_hp

        # First call for this battle: initialize state, no reward yet
        if self.prev_our_hp_pct is None or self.prev_opp_hp_pct is None:
            self.prev_our_hp_pct = our_hp
            self.prev_opp_hp_pct = opp_hp
            self.prev_hp_advantage = hp_advantage
            self.prev_our_fainted = self._count_fainted(current_battle, is_opponent=False)
            self.prev_opp_fainted = self._count_fainted(current_battle, is_opponent=True)
            self.prev_our_status = self._count_status_conditions(current_battle, is_opponent=False)
            self.prev_opp_status = self._count_status_conditions(current_battle, is_opponent=True)
            # hazards & screens
            self.prev_hazards_our = self._extract_hazards(current_battle.side_conditions)
            self.prev_hazards_opp = self._extract_hazards(current_battle.opponent_side_conditions)
            self.prev_screens_our = self._extract_screens(current_battle.side_conditions)
            self.prev_screens_opp = self._extract_screens(current_battle.opponent_side_conditions)
            # boost tracking
            self._snapshot_boosts(current_battle)
            # opp active
            self.prev_opp_active_id = self._get_opp_active_id(current_battle)
            return 0.0

        prev_hp_advantage = self.prev_hp_advantage

        # ---------- HP advantage change (micro signal) ----------
        hp_change = hp_advantage - prev_hp_advantage
        reward += hp_change * self.hp_weight

        # Momentum bonus for several favorable turns in a row
        if hp_change > 0:
            self.consecutive_favorable_turns += 1
            if self.consecutive_favorable_turns >= self.momentum_turns:
                reward += self.momentum_reward
        else:
            self.consecutive_favorable_turns = 0

        # ---------- DAMAGE CHUNKS (meso) ----------
        damage_dealt = max(0.0, self.prev_opp_hp_pct - opp_hp)
        damage_taken = max(0.0, self.prev_our_hp_pct - our_hp)

        # Big hit on opponent → reward
        if damage_dealt > 0.30:
            reward += self.type_effectiveness_reward
        # Only tiny chip → probably bad move (NVE-ish / stall w/ no progress)
        elif 0.0 < damage_dealt < 0.10:
            reward += self.type_effectiveness_penalty

        # Being chunked hard is quite bad
        if damage_taken > 0.30:
            reward += -abs(self.type_effectiveness_penalty)

        # ---------- KO rewards (macro) ----------
        our_fainted = self._count_fainted(current_battle, is_opponent=False)
        opp_fainted = self._count_fainted(current_battle, is_opponent=True)

        our_kos = max(0, our_fainted - self.prev_our_fainted)
        opp_kos = max(0, opp_fainted - self.prev_opp_fainted)

        # They lose mons → great
        if opp_kos > 0:
            reward += opp_kos * self.ko_reward
        # We lose mons → bad
        if our_kos > 0:
            reward += our_kos * self.ko_penalty

        # ---------- STATUS ----------
        our_status = self._count_status_conditions(current_battle, is_opponent=False)
        opp_status = self._count_status_conditions(current_battle, is_opponent=True)

        if our_status - self.prev_our_status > 0:
            reward += self.status_penalty
        if opp_status - self.prev_opp_status > 0:
            reward += self.status_reward

        # =====================================================================
        #                       STAT BOOSTS & DEBUFFS
        # =====================================================================
        reward += self._stat_boost_debuff_reward(current_battle, damage_dealt)

        # =====================================================================
        #                            HAZARDS & SCREENS
        # =====================================================================
        reward += self._hazard_and_screen_reward(current_battle, action_was_switch)

        # =====================================================================
        #                         GOOD SWITCH REWARD (OFF/DEF)
        # =====================================================================
        if action_was_switch:
            reward += self._good_switch_reward(current_battle, damage_dealt, damage_taken)

        # =====================================================================
        #                         FORCED SWITCH REWARD
        # =====================================================================
        reward += self._forced_switch_reward(current_battle)

        # =====================================================================
        #                         WEATHER / TERRAIN
        # =====================================================================
        reward += self._small_weather_terrain_reward(current_battle)

        # ---------- Turn penalty ----------
        reward += self.turn_penalty

        # ---------- UPDATE INTERNAL STATE FOR NEXT TURN ----------
        self.prev_our_hp_pct = our_hp
        self.prev_opp_hp_pct = opp_hp
        self.prev_hp_advantage = hp_advantage
        self.prev_our_fainted = our_fainted
        self.prev_opp_fainted = opp_fainted
        self.prev_our_status = our_status
        self.prev_opp_status = opp_status
        self.prev_hazards_our = self._extract_hazards(current_battle.side_conditions)
        self.prev_hazards_opp = self._extract_hazards(current_battle.opponent_side_conditions)
        self.prev_screens_our = self._extract_screens(current_battle.side_conditions)
        self.prev_screens_opp = self._extract_screens(current_battle.opponent_side_conditions)
        self._snapshot_boosts(current_battle)
        self.prev_opp_active_id = self._get_opp_active_id(current_battle)

        return reward

    # =============================
    # BOOST / DEBUFF REWARD
    # =============================

    def _snapshot_boosts(self, battle: Battle) -> None:
        """Store current boost stages for our active + opponent active."""
        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        if active is not None and active.boosts:
            self.prev_our_boosts = {
                "atk": active.boosts.get("atk", 0),
                "spa": active.boosts.get("spa", 0),
                "spe": active.boosts.get("spe", 0),
                "def": active.boosts.get("def", 0),
                "spd": active.boosts.get("spd", 0),
            }
        else:
            self.prev_our_boosts = {"atk": 0, "spa": 0, "spe": 0, "def": 0, "spd": 0}

        if opp is not None and opp.boosts:
            self.prev_opp_boosts = {
                "atk": opp.boosts.get("atk", 0),
                "spa": opp.boosts.get("spa", 0),
                "spe": opp.boosts.get("spe", 0),
                "def": opp.boosts.get("def", 0),
                "spd": opp.boosts.get("spd", 0),
            }
        else:
            self.prev_opp_boosts = {"atk": 0, "spa": 0, "spe": 0, "def": 0, "spd": 0}

    def _stat_boost_debuff_reward(self, battle: Battle, damage_dealt: float) -> float:
        """
        Reward boosting our own stats, penalize losing them.
        Reward lowering opponent stats, penalize letting them boost.
        Uses diminishing returns as stages grow.
        """
        reward = 0.0
        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        # Stats we track + base values
        tracked_stats = {
            "atk": 2.0,
            "spa": 2.0,
            "spe": 3.0,
            "def": 1.5,
            "spd": 1.5,
        }

        # ----- OUR BOOSTS / DROPS -----
        if active is not None:
            boosts = active.boosts or {}
            for stat_name, base_reward in tracked_stats.items():
                old_val = self.prev_our_boosts.get(stat_name, 0)
                new_val = boosts.get(stat_name, 0)
                delta = new_val - old_val

                if delta != 0:
                    # diminishing multiplier
                    diminishing = 1.0 / (1.0 + abs(old_val))

                    if delta > 0:
                        # we boosted
                        reward += delta * base_reward * diminishing
                        # slight bonus if we purely set up (no damage dealt)
                        if damage_dealt == 0.0:
                            reward += 0.4
                    else:
                        # we got our stats lowered
                        reward += delta * base_reward * diminishing  # negative

        # ----- OPPONENT BOOSTS / DROPS -----
        if opp is not None:
            boosts_o = opp.boosts or {}
            for stat_name, base_reward in tracked_stats.items():
                old_val = self.prev_opp_boosts.get(stat_name, 0)
                new_val = boosts_o.get(stat_name, 0)
                delta = new_val - old_val

                if delta != 0:
                    diminishing = 1.0 / (1.0 + abs(old_val))

                    if delta > 0:
                        # they boosted → bad for us
                        reward -= delta * base_reward * diminishing
                    else:
                        # they got debuffed → good
                        reward += (-delta) * base_reward * diminishing

        return reward

    # =============================
    # HAZARDS & SCREENS REWARD
    # =============================

    def _hazard_and_screen_reward(self, battle: Battle, action_was_switch: bool) -> float:
        reward = 0.0

        # Hazards on our side vs opp side
        our_haz = self._extract_hazards(battle.side_conditions)
        opp_haz = self._extract_hazards(battle.opponent_side_conditions)

        # ---- Setting / losing hazards ----
        # Opponent side hazards (we set them)
        delta_opp_sr = opp_haz["sr"] - self.prev_hazards_opp["sr"]
        delta_opp_spikes = opp_haz["spikes"] - self.prev_hazards_opp["spikes"]
        delta_opp_tspikes = opp_haz["tspikes"] - self.prev_hazards_opp["tspikes"]

        if delta_opp_sr > 0:
            reward += self.hazard_sr_set_reward
        if delta_opp_spikes > 0:
            reward += delta_opp_spikes * self.hazard_spikes_layer_reward
        if delta_opp_tspikes > 0:
            reward += delta_opp_tspikes * self.hazard_tspikes_layer_reward

        # If our hazards get removed from their side
        if delta_opp_sr < 0 or delta_opp_spikes < 0 or delta_opp_tspikes < 0:
            reward += self.hazard_removed_penalty

        # Our side hazards (they set them)
        delta_our_sr = our_haz["sr"] - self.prev_hazards_our["sr"]
        delta_our_spikes = our_haz["spikes"] - self.prev_hazards_our["spikes"]
        delta_our_tspikes = our_haz["tspikes"] - self.prev_hazards_our["tspikes"]

        if delta_our_sr > 0 or delta_our_spikes > 0 or delta_our_tspikes > 0:
            reward += self.hazard_removed_penalty  # they added hazards on us

        # If we removed hazards on our side (via Defog, Spin, etc.)
        if delta_our_sr < 0 or delta_our_spikes < 0 or delta_our_tspikes < 0:
            reward += self.hazard_remove_reward

        # ---- Extra penalty for switching into hazards ----
        if action_was_switch:
            total_layers_our = (
                our_haz["sr"]
                + our_haz["spikes"]
                + our_haz["tspikes"]
                + our_haz["webs"]
            )
            if total_layers_our > 0:
                # We already get HP loss from hp_change; this is just a small extra
                reward -= self.hazard_switch_penalty_scale * total_layers_our

        # ---- Screens / Veil ----
        our_screens = self._extract_screens(battle.side_conditions)
        opp_screens = self._extract_screens(battle.opponent_side_conditions)

        def screen_delta_reward(curr, prev, sign_for_us: float) -> float:
            r = 0.0
            for key in ["reflect", "lscreen", "veil"]:
                d = curr[key] - prev[key]
                if d > 0:
                    r += self.screen_set_reward * sign_for_us
                elif d < 0:
                    r += self.screen_lost_penalty * sign_for_us
            return r

        # We gain / lose screens → good/bad
        reward += screen_delta_reward(our_screens, self.prev_screens_our, +1.0)
        # They gain / lose screens → opposite sign
        reward += screen_delta_reward(opp_screens, self.prev_screens_opp, -1.0)

        return reward

    # =============================
    # GOOD SWITCH REWARD
    # =============================

    def _good_switch_reward(self, battle: Battle, damage_dealt: float, damage_taken: float) -> float:
        """
        Reward smart switches:
        - into faster mon
        - into SE STAB coverage
        - into resist/immunity to last move
        Also mildly penalize "terrible" switches (4x weak into their last type).
        """
        reward = 0.0

        # Base tempo cost
        reward += self.switch_penalty

        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        if active is None or opp is None:
            return reward

        # 1) Speed advantage
        speed_adv = False
        try:
            if active.stats and opp.stats:
                speed_adv = active.stats.get("spe", 0) > opp.stats.get("spe", 0)
        except Exception:
            speed_adv = False

        # 2) Offensive coverage (SE, STAB)
        best_mult = 1.0
        has_se_stab = False
        bad_offensive_type = True  # assume bad until proven otherwise

        for move in active.moves.values():
            mult = self._type_effectiveness(move, opp)
            if mult > best_mult:
                best_mult = mult
            if mult >= 1.0:
                bad_offensive_type = False
            try:
                if (
                    mult >= 2.0
                    and move.type is not None
                    and (
                        move.type == active.type_1
                        or (active.type_2 is not None and move.type == active.type_2)
                    )
                ):
                    has_se_stab = True
            except Exception:
                pass

        good_type = best_mult >= 2.0
        bad_type = best_mult < 1.0

        # 3) Defensive quality vs opponent's *last* move
        last_opp_move = getattr(battle, "opponent_last_used_move", None)
        resist_bonus = 0.0
        if last_opp_move is not None:
            mult_taken = self._type_effectiveness(last_opp_move, active)
            # Immune switch-in → godlike
            if mult_taken == 0.0:
                resist_bonus += 6.0
            # Resist → solid
            elif mult_taken < 1.0:
                resist_bonus += 3.0
            # Super effective into our switch → bad
            elif mult_taken > 1.0:
                resist_bonus -= 4.0

        # 4) Combine into switch score
        switch_score = 0.0

        # Offensive pressure:
        if has_se_stab and speed_adv:
            switch_score += 6.0   # faster & SE STAB → big pressure
        elif has_se_stab:
            switch_score += 4.0
        elif good_type:
            switch_score += 2.0
        elif bad_type and bad_offensive_type:
            switch_score -= 2.0

        # Speed-only advantage:
        if speed_adv and not has_se_stab and not good_type:
            switch_score += 2.0

        # Add defensive bonus from resist/immunity
        switch_score += resist_bonus

        # If we switched and took basically no damage, it's likely a safe pivot
        if damage_taken == 0.0:
            switch_score += 1.0
        elif damage_taken > 0.3:
            # we got chunked immediately after switching in
            switch_score -= 2.0

        # Clamp to keep reward sane
        switch_score = max(-10.0, min(10.0, switch_score))

        reward += switch_score
        return reward

    # =============================
    # FORCED SWITCH HEURISTIC
    # =============================

    def _get_opp_active_id(self, battle: Battle):
        opp = battle.opponent_active_pokemon
        if opp is None:
            return None
        try:
            return to_id_str(opp.species)
        except Exception:
            return None

    def _forced_switch_reward(self, battle: Battle) -> float:
        """
        Reward situations where the opponent's active mon changes
        without us KOing it (heuristic for "we forced a switch").

        We also give a small extra bonus if hazards are up on their side.
        """
        reward = 0.0
        current_id = self._get_opp_active_id(battle)
        if current_id is None or self.prev_opp_active_id is None:
            return 0.0

        # If the active species changed and we didn't KO something this turn,
        # we likely forced them out.
        opp_fainted = self._count_fainted(battle, is_opponent=True)
        if current_id != self.prev_opp_active_id and opp_fainted == self.prev_opp_fainted:
            reward += self.forced_switch_reward

            # Extra synergy bonus if hazards are up on their side
            opp_haz = self._extract_hazards(battle.opponent_side_conditions)
            total_layers_opp = (
                opp_haz["sr"]
                + opp_haz["spikes"]
                + opp_haz["tspikes"]
                + opp_haz["webs"]
            )
            if total_layers_opp > 0:
                reward += self.forced_switch_hazard_bonus * total_layers_opp

        return reward

    # =============================
    # TERMINAL REWARD
    # =============================

    def compute_terminal_reward(self, battle: Battle) -> float:
        """
        Compute terminal reward when battle ends.
        Large signal to strongly prefer winning games.
        """
        if battle.won:
            return self.terminal_win
        elif battle.lost:
            return self.terminal_loss
        else:
            return self.terminal_draw

    # =============================
    # RESET BETWEEN BATTLES
    # =============================

    def reset(self):
        """Reset internal state for a new battle."""
        self.prev_hp_advantage = 0.0
        self.prev_our_hp_pct = None
        self.prev_opp_hp_pct = None
        self.prev_our_fainted = 0
        self.prev_opp_fainted = 0
        self.prev_our_status = 0
        self.prev_opp_status = 0

        self.prev_our_boosts = {"atk": 0, "spa": 0, "spe": 0, "def": 0, "spd": 0}
        self.prev_opp_boosts = {"atk": 0, "spa": 0, "spe": 0, "def": 0, "spd": 0}

        self.prev_hazards_our = {"sr": 0, "spikes": 0, "tspikes": 0, "webs": 0}
        self.prev_hazards_opp = {"sr": 0, "spikes": 0, "tspikes": 0, "webs": 0}
        self.prev_screens_our = {"reflect": 0, "lscreen": 0, "veil": 0}
        self.prev_screens_opp = {"reflect": 0, "lscreen": 0, "veil": 0}

        self.prev_weather = None
        self.prev_terrain = None
        self.prev_opp_active_id = None

        self.consecutive_favorable_turns = 0
