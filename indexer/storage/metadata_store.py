"""SQLite metadata store for colors, env, garments, objects, and patch paths."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional


class MetadataStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS images(
                id TEXT PRIMARY KEY,
                path TEXT,
                env_label TEXT,
                env_conf REAL,
                env_embed TEXT,
                patch_path TEXT,
                brightness REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS colors(
                image_id TEXT,
                color TEXT,
                prob REAL,
                PRIMARY KEY(image_id, color)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS garments(
                image_id TEXT,
                tag TEXT,
                score REAL,
                PRIMARY KEY(image_id, tag)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS objects(
                image_id TEXT,
                label TEXT,
                score REAL,
                PRIMARY KEY(image_id, label)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_env_label ON images(env_label)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_color ON colors(color)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_object_label ON objects(label)")
        conn.commit()
        conn.close()

    def upsert_image(
        self,
        image_id: str,
        path: str,
        env_label: str,
        env_conf: float,
        env_embed: Optional[list],
        patch_path: str,
        brightness: float,
        colors: Dict[str, float],
        garments: Dict[str, float],
        objects: Dict[str, float],
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO images(id, path, env_label, env_conf, env_embed, patch_path, brightness)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                path=excluded.path,
                env_label=excluded.env_label,
                env_conf=excluded.env_conf,
                env_embed=excluded.env_embed,
                patch_path=excluded.patch_path,
                brightness=excluded.brightness
            """,
            (image_id, path, env_label, env_conf, json.dumps(env_embed) if env_embed else None, patch_path, brightness),
        )
        for color, prob in colors.items():
            cur.execute(
                """
                INSERT INTO colors(image_id, color, prob)
                VALUES(?, ?, ?)
                ON CONFLICT(image_id, color) DO UPDATE SET prob=excluded.prob
                """,
                (image_id, color, prob),
            )
        for tag, score in garments.items():
            cur.execute(
                """
                INSERT INTO garments(image_id, tag, score)
                VALUES(?, ?, ?)
                ON CONFLICT(image_id, tag) DO UPDATE SET score=excluded.score
                """,
                (image_id, tag, score),
            )
        for label, score in objects.items():
            cur.execute(
                """
                INSERT INTO objects(image_id, label, score)
                VALUES(?, ?, ?)
                ON CONFLICT(image_id, label) DO UPDATE SET score=excluded.score
                """,
                (image_id, label, score),
            )
        conn.commit()
        conn.close()

    def fetch(self, image_id: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, path, env_label, env_conf, env_embed, patch_path, brightness FROM images WHERE id=?", (image_id,))
        row = cur.fetchone()
        if row is None:
            conn.close()
            return None
        data = {
            "id": row[0],
            "path": row[1],
            "env_label": row[2],
            "env_conf": row[3],
            "env_embed": json.loads(row[4]) if row[4] else None,
            "patch_path": row[5],
            "brightness": row[6],
        }
        cur.execute("SELECT color, prob FROM colors WHERE image_id=?", (image_id,))
        data["colors"] = {c: p for c, p in cur.fetchall()}
        cur.execute("SELECT tag, score FROM garments WHERE image_id=?", (image_id,))
        data["garments"] = {t: s for t, s in cur.fetchall()}
        cur.execute("SELECT label, score FROM objects WHERE image_id=?", (image_id,))
        data["objects"] = {t: s for t, s in cur.fetchall()}
        conn.close()
        return data
