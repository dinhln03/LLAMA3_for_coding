def up(cursor, bot):
    cursor.execute(
        """
    CREATE TABLE timeouts(
        id SERIAL PRIMARY KEY,
        active BOOL DEFAULT TRUE NOT NULL,
        user_id TEXT REFERENCES "user"(discord_id) ON DELETE CASCADE NOT NULL,
        issued_by_id TEXT REFERENCES "user"(discord_id) ON DELETE SET NULL,
        unbanned_by_id TEXT REFERENCES "user"(discord_id) ON DELETE SET NULL,
        ban_reason TEXT,
        unban_reason TEXT,
        until TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL
    )
    """
    )
