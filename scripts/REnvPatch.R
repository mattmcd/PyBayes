for (nm in c("system", "system2")) {
  # See https://youtrack.jetbrains.com/issue/R-1543/R-Console-function-calls-fail-complain-about-excessive-arguments#focus=Comments-27-10291837.0-0
  env <- baseenv()
  fun <- get(nm, envir = env)
  bd <- body(fun)
  if (length(bd[[length(bd)]][[2L]]) == 5L) {
    # a 4-argument call to system()
    bd[[length(bd)]][[2L]][[5L]] <- NULL
    body(fun) <- bd
    unlockBinding(nm, env)
    assign(nm, fun, envir = env)
    lockBinding(nm, env)
    cat("v patched ", nm, ".\n", sep = "")
  } else {
    cat("i", nm, " was already patched.\n", sep = "")
  }
}