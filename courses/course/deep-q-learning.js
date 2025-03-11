/**
 * Deep Q-Learning Page JavaScript
 * This file contains all the interactive functionality for the Deep Q-Learning page
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all interactive components
    initSmoothScrolling();
    initCodeHighlighting();
    initPseudocodeOverlay();
});

/**
 * Initialize smooth scrolling for anchor links
 */
function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
}

/**
 * Initialize code highlighting 
 * Highlights code examples when they are directly accessed via URL hash
 */
function initCodeHighlighting() {
    document.querySelectorAll('.code-example').forEach(codeExample => {
        const id = codeExample.getAttribute('id');
        if (window.location.hash === '#' + id) {
            codeExample.scrollIntoView();
            codeExample.style.backgroundColor = '#f0f8ff';
            setTimeout(() => {
                codeExample.style.backgroundColor = '';
            }, 1500);
        }
    });
}

/**
 * Initialize the pseudocode overlay system
 * This replaces the default code section visibility toggle
 */
function initPseudocodeOverlay() {
    // Get all necessary elements
    const codeToggleButtons = document.querySelectorAll('.code-toggle-button');
    const overlay = document.getElementById('pseudocodeOverlay');
    const pseudocodeContent = document.getElementById('pseudocodeContent');
    const closeButton = document.getElementById('closePseudocode');
    
    // Add click event to all toggle buttons
    codeToggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const codeSection = document.getElementById(targetId);
            
            // Get the content from the original code section
            const content = codeSection.innerHTML;
            
            // Set content to the overlay
            pseudocodeContent.innerHTML = content;
            
            // Show overlay
            overlay.classList.add('active');
            
            // Prevent scrolling on the body
            document.body.style.overflow = 'hidden';
            
            // Keep the button text consistent
            if (!this.textContent.startsWith('Show')) {
                this.textContent = 'Show ' + this.textContent.split(' ').slice(1).join(' ');
            }
        });
    });
    
    // Close overlay when clicking the close button
    closeButton.addEventListener('click', function() {
        overlay.classList.remove('active');
        document.body.style.overflow = 'auto';
    });
    
    // Close overlay when clicking outside the content
    overlay.addEventListener('click', function(e) {
        if (e.target === overlay) {
            overlay.classList.remove('active');
            document.body.style.overflow = 'auto';
        }
    });
    
    // Close on escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && overlay.classList.contains('active')) {
            overlay.classList.remove('active');
            document.body.style.overflow = 'auto';
        }
    });
}