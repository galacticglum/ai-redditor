$(document).ready(function() {
    $('body').show();

    // Redacted text effect
    $('.redacted').hover(function() {
        $(this).toggleClass('redacted-remove-hover');
    });
    
    // Check if collapse-toggle should be active (is the source text overflowing?)
    const postBodyTextElement = $($('#collapse-toggle').data('expand-target'));
    const isReadmoreVisible = postBodyTextElement.width() >= postBodyTextElement[0].scrollWidth;
    $('#collapse-toggle').toggle(isReadmoreVisible);

    // Post body expand ("read more") button
    $('#collapse-toggle').on('click', function(e) {
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
        $('#collapse-toggle').data('expanded', expanded)
    });
});