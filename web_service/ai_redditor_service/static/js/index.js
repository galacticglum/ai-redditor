$(document).ready(function() {
    $('body').show();

    // Redacted text effect
    $('.redacted').hover(function() {
        $(this).toggleClass('redacted-remove-hover');
    });
    
    // Check if collapse-toggle should be active (is the source text overflowing?)
    const postBodyTextElement = $($('#collapse-toggle').data('expand-target'));
    const isReadmoreVisible = postBodyTextElement[0].scrollHeight > postBodyTextElement[0].clientHeight;
    $('#collapse-toggle').toggle(isReadmoreVisible);

    // Post body expand ("read more") button
    $('#collapse-toggle').on('click', function() {
        // Toggling truncation works by changing the display CSS attribute.
        var expanded = $(this).data('expanded');
        if (expanded === undefined) {
            expanded = false;
        }

        expanded = !expanded;
        const expandTargetElement = $($(this).data('expand-target'));
        const expandLabelElement = $($(this).data('expand-label-target'));
        if (expanded) {
            expandTargetElement.addClass('d-block');
            expandLabelElement.text('Read less');
        } else {
            expandTargetElement.removeClass('d-block');
            expandLabelElement.text('Read more');
        }

        // Toggle padding on the footer
        $('.mastfoot').toggleClass('pb-3');
        $('#collapse-toggle').data('expanded', expanded);
    });

    function togglePostViewVisibility(isPostViewActive) {
        const generateButtonElement = $('#generate-button');
        const postViewElement = $(generateButtonElement.data('post-view-target'));
        const generateViewElement = $(generateButtonElement.data('generate-view-target'));

        if (isPostViewActive) {
            postViewElement.removeClass('d-none');
            generateViewElement.addClass('d-none');
        } else {
            postViewElement.addClass('d-none');
            generateViewElement.removeClass('d-none');
        }
    }

    // Handle changing the view between generated post and generate new post
    $('#generate-button').on('click', function() {
        togglePostViewVisibility(false);
    });

    $('#cancel-generate-button').on('click', function() {
        togglePostViewVisibility(true);
    });

    // Initialize copy to clipboard action for permalink button
    new ClipboardJS('#permalink');

    // Enable tooltips everywhere
    $('[data-toggle="tooltip"]').tooltip();
    // $('#permalink').on('mouseleave', function() {
    //     $(this).tooltip('hide');
    // });

    // Disable permalink link action
    $('#permalink').on('click', function(e) {
        const element = $(this);
        element.tooltip('show');
        setTimeout(function() {
            element.tooltip('hide');
        }, 500);

        e.preventDefault();
    });
});
