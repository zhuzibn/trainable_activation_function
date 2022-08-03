function [y_pos_max,y_neg_max]=find_max_variable(y,y_pos_max,y_neg_max)
y_pos_max_tmp=max(y(y>0));
if ~isempty(y_pos_max_tmp)
    y_pos_max=max(y_pos_max_tmp,y_pos_max);
end

y_neg_max_tmp=max(-y(y<0));
if ~isempty(y_neg_max_tmp)
    y_neg_max=max(y_neg_max_tmp,y_neg_max);
end