// Set default role on load
document.addEventListener('DOMContentLoaded', () => {
    selectRole('student');
});

// Function to handle role selection
function selectRole(roleName) {
    // 1. Get all role cards
    const cards = document.querySelectorAll('.role-card');
    
    // 2. Remove 'active' class from all
    cards.forEach(card => {
        card.classList.remove('active');
    });

    // 3. Add 'active' class to the clicked one
    const selected = document.getElementById(`role-${roleName}`);
    if (selected) {
        selected.classList.add('active');
    }
}

// Simple form handler preventing default refresh
function handleLogin(e) {
    e.preventDefault();
    
    const email = document.querySelector('input[type="email"]').value;
    const password = document.querySelector('input[type="password"]').value;
    
    // Add your actual login logic here
    console.log("Login attempted for:", email);
    
    // Visual feedback button animation
    const btn = document.querySelector('.submit-btn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Logging in...';
    
    setTimeout(() => {
        btn.innerHTML = originalText;
        alert("Login functionality would trigger here.");
    }, 1500);
}
