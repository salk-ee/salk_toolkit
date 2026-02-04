# DEV-37: Secrets-based auth bypass

**Last Updated**: 2026-02-04  
**Status**: ⏳ In Progress  
**Module**: Dashboard  
**Tags**: `#feature`, `#auth`, `#dev`  
**Dependencies**: None  

## Overview

Add a secrets-driven local dev login bypass that constructs a user from `auth.bypass_user` and skips interactive auth flows.

## Problem Context

- Local dev needs a zero-click login path driven by `secrets.toml`.
- Intended use cases
  - Single-developer dashboard debugging without Frontegg or legacy auth.
  - Providing credential-free access to AI coding agents
- Technical constraints and requirements
  - Must not require new external services.
  - Should keep existing `SalkDashboardBuilder` integration points unchanged for page rendering.
- Integration points with existing systems
  - `SalkDashboardBuilder` auth manager selection in `salk_toolkit/dashboard.py`.
  - Streamlit secrets access via `st.secrets`.

## Requirements

**Important context functions/files**
- `salk_toolkit/dashboard.py` for `SalkDashboardBuilder` and existing `UserAuthenticationManager` implementations.
- `tests/test_dashboard.py` for auth-related unit tests and patterns.
- Domain rules: `dashboard.mdc` for dashboard behavior expectations, `salk_toolkit.mdc` for typing/docstring/testing conventions.

**Files to Create/Modify:**
- Modify `salk_toolkit/dashboard.py` to add `BypassAuthenticationManager` and wire it into auth selection.
- Modify `tests/test_dashboard.py` to cover bypass behavior.

**Functionality:**
- If `st.secrets["auth"]["bypass_user"]` exists, use a new `BypassAuthenticationManager` instance.
- `bypass_user` fields override defaults; defaults are `name` is required, `lang="en"`, `organization="SALK"`, `group="admin"`.
- `uid` defaults to `name`.
- Bypass manager sets `authenticated=True` and returns the constructed user via `.user`.
- Admin-related methods (`add_user`, `change_user`, `delete_user`, `list_users`, `update_user`) use dummy implementations that are as close to no-op as possible.
  - One exception: for testing purposes, language (`sdb.user["lang"]`) should be changeable during session (stored in st.session if changed)
- Org whitelist checks still apply in bypass mode.
- If `bypass_user.name` is missing, raise a clear error and do not fall back to other auth flows.

**Architecture:**
- Follow existing auth manager abstraction (`UserAuthenticationManager`) and keep `SalkDashboardBuilder` as the orchestrator.

## Implementation Plan

### Foundation Setup

- [x] Add `BypassAuthenticationManager` class in `salk_toolkit/dashboard.py` with required overrides and defaults.

### Core Development

- [x] Update `SalkDashboardBuilder.__init__` to select bypass manager when `auth.bypass_user` is present.
- [x] Ensure translation language uses `lang` from bypass user with fallback to `en`.

### Integration & Testing

- [x] Add unit tests for bypass user construction, defaulting, and auth selection.

## Definition of Done

- [ ] `BypassAuthenticationManager` is used when `auth.bypass_user` exists in secrets.
- [ ] Bypass user defaults and required fields behave as specified.
- [ ] Dummy admin methods are present and do not perform side effects.
- [ ] Tests for bypass behavior pass.

## Implementation Notes

- Stored bypass user in `st.session_state` to persist language updates in-session.

## Q&A
