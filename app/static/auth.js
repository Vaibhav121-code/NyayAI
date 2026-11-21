const authForm = document.querySelector("#authForm");
const authEmail = document.querySelector("#authEmail");
const authPassword = document.querySelector("#authPassword");
const confirmPasswordField = document.querySelector("#confirmPasswordField");
const authPasswordConfirm = document.querySelector("#authPasswordConfirm");
const authMessage = document.querySelector("#authMessage");
const authSubmit = document.querySelector("#authSubmit");
const authTitle = document.querySelector("#authTitle");
const authDescription = document.querySelector("#authDescription");
const showLogin = document.querySelector("#showLogin");
const showSignup = document.querySelector("#showSignup");

let authMode = "login";

function setAuthMode(mode) {
  authMode = mode;
  const isSignup = mode === "signup";
  authTitle.textContent = isSignup ? "Create your account" : "Welcome back";
  authDescription.textContent = isSignup
    ? "Create an account to use Nyay AI."
    : "Log in to continue to Nyay AI.";
  authSubmit.textContent = isSignup ? "Create account" : "Log in";
  confirmPasswordField.hidden = !isSignup;
  authPassword.autocomplete = isSignup ? "new-password" : "current-password";
  authPasswordConfirm.required = isSignup;
  showLogin.setAttribute("aria-pressed", String(!isSignup));
  showSignup.setAttribute("aria-pressed", String(isSignup));
  authMessage.hidden = true;
  authMessage.textContent = "";
}

showLogin.addEventListener("click", () => setAuthMode("login"));
showSignup.addEventListener("click", () => setAuthMode("signup"));

fetch("/api/auth/me")
  .then(async response => {
    if (!response.ok) throw new Error("Unable to check your sign-in status.");
    const result = await response.json();
    if (result.authenticated) window.location.replace("/");
  })
  .catch(error => {
    authMessage.textContent = error.message;
    authMessage.hidden = false;
  });

authForm.addEventListener("submit", async event => {
  event.preventDefault();
  if (authMode === "signup" && authPassword.value !== authPasswordConfirm.value) {
    authMessage.textContent = "The passwords do not match.";
    authMessage.hidden = false;
    authPasswordConfirm.focus();
    return;
  }

  authSubmit.disabled = true;
  authSubmit.textContent = authMode === "signup" ? "Creating account…" : "Logging in…";
  authMessage.hidden = true;
  try {
    const response = await fetch(`/api/auth/${authMode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: authEmail.value.trim(),
        password: authPassword.value,
      }),
    });
    const result = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(result.detail)
        ? result.detail.map(item => item.msg).join(" ")
        : result.detail;
      throw new Error(detail || "Authentication failed.");
    }
    window.location.replace("/");
  } catch (error) {
    authMessage.textContent = error.message;
    authMessage.hidden = false;
  } finally {
    authSubmit.disabled = false;
    authSubmit.textContent = authMode === "signup" ? "Create account" : "Log in";
  }
});
