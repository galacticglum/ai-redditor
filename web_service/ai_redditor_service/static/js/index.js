$(document).ready(function() {
    $('.redacted').hover(function() {
        $('.redacted').toggleClass('redacted-remove-hover');
    });
    
    $('#collapse-toggle').on('click', function(e) {
        // Toggling truncation works by changing the display CSS attribute.
        var expanded = $('#collapse-toggle').data('expanded');
        if (expanded === undefined) {
            expanded = false;
        }

        expanded = !expanded;
        const expandTargetElement = $($('#collapse-toggle').data('expand-target'));
        const expandLabelElement = $($('#collapse-toggle').data('expand-label-target'));
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
