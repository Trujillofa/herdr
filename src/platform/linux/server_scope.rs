use std::io;
use std::process::{Command, Stdio};

const CGROUP_FS_ROOT: &str = "/sys/fs/cgroup";

/// Called after exec, before logging, worker threads, or pane creation.
///
/// `setsid()` alone leaves a detached server in the launching terminal's
/// systemd scope. When that terminal exits, or systemd-oomd picks the scope,
/// systemd kills every process in it, including the server and its panes.
/// Move a detached server into its own `herdr-server*.scope` that oomd omits.
pub(crate) fn prepare_server_process(handoff_import: bool) -> io::Result<bool> {
    // A replacement inherits its source's scope; direct `herdr server` runs
    // stay where the user started them.
    if handoff_import || !super::super::current_process_is_detached_server_daemon() {
        return Ok(false);
    }
    let cgroup = std::fs::read_to_string("/proc/self/cgroup")?;
    let Some(current) = unified_cgroup_path(&cgroup) else {
        return Ok(false);
    };
    if !is_foreign_user_scope(current) {
        return Ok(false);
    }
    move_into_oom_omitted_scope(&server_scope_unit(), std::process::id())?;
    Ok(true)
}

fn move_into_oom_omitted_scope(unit: &str, pid: u32) -> io::Result<()> {
    // A previous server with the same session name may have left a failed unit.
    let _ = Command::new("systemctl")
        .args(["--user", "reset-failed", unit])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    if !user_unit_is_active(unit) {
        let output = Command::new("busctl")
            .args(start_transient_scope_args(unit, pid))
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()?;
        if output.status.success() {
            return Ok(());
        }
        if !user_unit_is_active(unit) {
            return Err(io::Error::other(format!(
                "StartTransientUnit {unit}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
    }
    // The unit already exists (for example a stale scope that still holds a
    // pane): join it instead of creating a second one.
    let output = Command::new("systemctl")
        .args(["--user", "show", "-p", "ControlGroup", "--value", unit])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()?;
    let group = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !output.status.success() || group.is_empty() {
        return Err(io::Error::other(format!("{unit} has no control group")));
    }
    std::fs::write(
        format!("{CGROUP_FS_ROOT}{group}/cgroup.procs"),
        format!("{pid}\n"),
    )
}

fn user_unit_is_active(unit: &str) -> bool {
    Command::new("systemctl")
        .args(["--user", "is-active", "--quiet", unit])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn start_transient_scope_args(unit: &str, pid: u32) -> Vec<String> {
    [
        "--user",
        "call",
        "org.freedesktop.systemd1",
        "/org/freedesktop/systemd1",
        "org.freedesktop.systemd1.Manager",
        "StartTransientUnit",
        "ssa(sv)a(sa(sv))",
        unit,
        "fail",
        "2",
        "PIDs",
        "au",
        "1",
        &pid.to_string(),
        "ManagedOOMPreference",
        "s",
        "omit",
        "0",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn unified_cgroup_path(contents: &str) -> Option<&str> {
    contents
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .map(str::trim)
        .filter(|path| !path.is_empty())
}

/// A scope owned by another app (a terminal) inside the user's systemd manager.
fn is_foreign_user_scope(path: &str) -> bool {
    let mut parts = path.split('/').filter(|part| !part.is_empty());
    let in_user_manager = parts.any(|part| part.starts_with("user@") && part.ends_with(".service"));
    in_user_manager
        && path
            .rsplit('/')
            .next()
            .is_some_and(|leaf| leaf.ends_with(".scope") && !leaf.starts_with("herdr-server"))
}

fn server_scope_unit() -> String {
    match std::env::var("HERDR_SESSION") {
        Ok(name) if !name.is_empty() && name != "default" => {
            let safe: String = name
                .chars()
                .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
                .collect();
            format!("herdr-server-{safe}.scope")
        }
        _ => "herdr-server.scope".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KITTY_SCOPE: &str = "/user.slice/user-1000.slice/user@1000.service/app.slice/app-graphical.slice/kitty-3533997-0.scope";

    #[test]
    fn reads_unified_cgroup_line() {
        assert_eq!(
            unified_cgroup_path(&format!("12:perf_event:/\n0::{KITTY_SCOPE}\n")),
            Some(KITTY_SCOPE)
        );
        assert_eq!(unified_cgroup_path("1:name=systemd:/\n"), None);
    }

    #[test]
    fn only_foreign_user_scopes_are_left() {
        assert!(is_foreign_user_scope(KITTY_SCOPE));
        assert!(!is_foreign_user_scope(
            "/user.slice/user-1000.slice/user@1000.service/herdr-server.scope"
        ));
        assert!(!is_foreign_user_scope(
            "/user.slice/user-1000.slice/user@1000.service/app.slice"
        ));
        assert!(!is_foreign_user_scope(
            "/user.slice/user-1000.slice/session-2.scope"
        ));
        assert!(!is_foreign_user_scope("/system.slice/herdr.service"));
    }

    #[test]
    fn transient_scope_is_omitted_from_oomd() {
        let args = start_transient_scope_args("herdr-server.scope", 22444);
        let position = |value: &str| args.iter().position(|arg| arg == value).unwrap();
        assert_eq!(
            args[position("StartTransientUnit") + 2],
            "herdr-server.scope"
        );
        assert_eq!(args[position("PIDs") + 3], "22444");
        assert_eq!(
            &args[position("ManagedOOMPreference")..=position("ManagedOOMPreference") + 2],
            ["ManagedOOMPreference", "s", "omit"]
        );
    }
}
