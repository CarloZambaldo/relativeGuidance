% Initialize variables
img = zeros(500, 500, 3, 'uint8');
snake_position = [250, 250; 240, 250; 230, 250];
apple_position = [randi([1, 50])*10, randi([1, 50])*10];
score = 0;
prev_button_direction = 1;
button_direction = 1;
snake_head = [250, 250];

while true
    imshow(img);
    drawnow;
    
    img = zeros(500, 500, 3, 'uint8');
    
    % Display Apple
    rectangle('Position', [apple_position(1), apple_position(2), 10, 10], 'EdgeColor', 'r', 'LineWidth', 3);
    
    % Display Snake
    for i = 1:size(snake_position, 1)
        rectangle('Position', [snake_position(i, 1), snake_position(i, 2), 10, 10], 'EdgeColor', 'g', 'LineWidth', 3);
    end
    
    % Takes step after fixed time
    t_end = toc + 0.05;
    k = -1;
    
    while toc < t_end
        if k == -1
            k = get(gcf, 'CurrentKey');
        end
    end
    
    % Handle key inputs for direction
    if strcmp(k, 'a') && prev_button_direction ~= 1
        button_direction = 0;
    elseif strcmp(k, 'd') && prev_button_direction ~= 0
        button_direction = 1;
    elseif strcmp(k, 'w') && prev_button_direction ~= 2
        button_direction = 3;
    elseif strcmp(k, 's') && prev_button_direction ~= 3
        button_direction = 2;
    elseif strcmp(k, 'q')
        break;
    end
    
    prev_button_direction = button_direction;
    
    % Change the head position based on the button direction
    if button_direction == 1
        snake_head(1) = snake_head(1) + 10;
    elseif button_direction == 0
        snake_head(1) = snake_head(1) - 10;
    elseif button_direction == 2
        snake_head(2) = snake_head(2) + 10;
    elseif button_direction == 3
        snake_head(2) = snake_head(2) - 10;
    end
    
    % Increase Snake length on eating apple
    if isequal(snake_head, apple_position)
        [apple_position, score] = collision_with_apple(apple_position, score);
        snake_position = [snake_head; snake_position];
    else
        snake_position = [snake_head; snake_position(1:end-1, :)];
    end
    
    % On collision, kill the snake and print the score
    if collision_with_boundaries(snake_head) == 1 || collision_with_self(snake_position) == 1
        font = 1;
        img = zeros(500, 500, 3, 'uint8');
        text(140, 250, sprintf('Your Score is %d', score), 'FontSize', 15, 'Color', 'w');
        fi = imshow(img);
        fi.drawnow();
        break;
    end
end

% Function Definitions
function [apple_position, score] = collision_with_apple(apple_position, score)
    apple_position = [randi([1, 50])*10, randi([1, 50])*10];
    score = score + 1;
end

function result = collision_with_boundaries(snake_head)
    if snake_head(1) >= 500 || snake_head(1) < 0 || snake_head(2) >= 500 || snake_head(2) < 0
        result = 1;
    else
        result = 0;
    end
end

function result = collision_with_self(snake_position)
    snake_head = snake_position(1, :);
    if any(ismember(snake_position(2:end, :), snake_head, 'rows'))
        result = 1;
    else
        result = 0;
    end
end
