# Secrets-based dashboard auth bypass (PR #37)

**Modules:** `salk_toolkit/dashboard.py`, `tests/test_dashboard.py`

## Goal

Debugging a dashboard locally meant going through Frontegg or streamlit-authenticator, which
is both slow and impossible for an AI coding agent with no credentials. Putting an
`auth.bypass_user` table in `secrets.toml` now constructs a user directly and skips the
interactive login, without adding any external service or changing how pages render.

## Design

- **`_get_bypass_user_from_secrets() -> JsonDict | None`** reads
  `st.secrets["auth"]["bypass_user"]` and returns `None` when it is absent — that `None` is
  what leaves the normal auth selection intact. When present it must be a mapping and must
  carry a `name`; anything else raises. Everything else defaults: `uid = name`, `lang = "en"`,
  `organization = "SALK"`, `group = "admin"`. Extra keys in the secret pass through to the
  user dict untouched.

- **`BypassAuthenticationManager(UserAuthenticationManager)`** implements the existing auth
  abstraction rather than short-circuiting around it, so `SalkDashboardBuilder` stays the
  orchestrator and every page-rendering integration point is unchanged. `authenticated` is
  always `True`, `passwordless = True`, `uam_user()` reads the session, and `login_screen` /
  `logout_button` are no-ops.

- **Admin methods are stubs, deliberately close to no-op:** `add_user` returns `False`,
  `delete_user` and `update_user` do nothing, `list_users` returns just the bypass user. The
  one exception is `change_user`, which updates `lang` in the session — the language switcher
  is a real thing to test locally, and it is the only admin action with no persistence
  consequences.

- **`SalkDashboardBuilder.__init__`** picks this manager when the secret is present, ahead of
  the other auth managers. The organization whitelist check is *not* bypassed: a bypass user
  in a non-whitelisted organization is still refused.

## Implementation notes

- The user dict is seeded into `st.session_state["bypass_user"]` once and read from there
  afterwards, so an in-session language change survives Streamlit reruns while the secrets
  file remains the source of truth for everything else.
- A `bypass_user` without a `name` is a hard error, never a fallback to the interactive
  flow — a half-configured bypass that quietly renders a login screen is the failure mode
  worth being loud about.
