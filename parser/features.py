# move feature extractor to a separate module

# Configuration -> FeatureVector
def extract_features_eng(c):
    """Represent configuration as a feature vector to be consumed by a classifier
    (using feature templates from page 32 of 06_howto_implement.pdf and slide 16 of
    04_machine_learnning.pdf)
    """
    # todo I'd rather have them as objects and properties. not essential though
    # will comply with the format for now

    def form_of_buffer_front():
        return c.sentence[c.buffer[0]].form

    def pos_of_buffer_front():
        return c.sentence[c.buffer[0]].cpostag

    def form_of_stack_top():
        try:
            return c.sentence[c.stack[-1]].form
        except IndexError:
            return 'None'

    def pos_of_stack_top():
        try:
            return c.sentence[c.stack[-1]].cpostag
        except IndexError:
            return 'None'

    def pos_of_second_buffer_item():
        try:
            return c.sentence[c.buffer[1]].cpostag
        except IndexError:
            return 'None'

    def pos_of_second_stack_item():
        try:
            return c.sentence[c.stack[-2]].cpostag
        except IndexError:
            return 'None'

    def pos_of_leftmost_dep_of_buffer_front():
        try:
            feature = c.sentence[min({arc for arc in c.arcs if arc.h == c.buffer[0]}, key=lambda arc: arc.d).d].cpostag
        except ValueError:
            feature = 'None'
        return feature

    def pos_of_stack_top_and_pos_of_buffer_front():
        try:
            return pos_of_stack_top() + ' ' + pos_of_buffer_front()
        except IndexError:
            return 'None'

    def pos_of_stack_top_and_form_of_buffer_front():
        try:
            return pos_of_stack_top() + ' ' + form_of_buffer_front()
        except IndexError:
            return 'None'

    def form_of_stack_top_and_pos_of_buffer_front():
        try:
            return form_of_stack_top() + ' ' + pos_of_buffer_front()
        except IndexError:
            return 'None'

    def form_of_stack_top_and_form_of_buffer_front():
        try:
            return form_of_stack_top() + ' ' + form_of_buffer_front()
        except IndexError:
            return 'None'

    # extra features from Kuebler's book
    def lemma_stk0():
        try:
            return c.sentence[c.stack[-1]].lemma
        except IndexError:
            return 'None'

    def lemma_buf0():
        return c.sentence[c.buffer[0]].lemma

    def form_buf1():
        try:
            return c.sentence[c.buffer[1]].form
        except IndexError:
            return 'None'

    def pos_buf2():
        try:
            return c.sentence[c.buffer[2]].cpostag
        except IndexError:
            return 'None'

    def pos_buf3():
        try:
            return c.sentence[c.buffer[3]].cpostag
        except IndexError:
            return 'None'

    def deprel_rdep_stk0():  # todo can the rightmost dependency be on the left?
        try:
            feature = c.sentence[max(
                {arc for arc in c.arcs if arc.h == c.stack[-1]}, key=lambda arc: arc.d).d].deprel
        except ValueError:
            feature = 'None'
        except IndexError:
            feature = 'None'
        return feature

    def deprel_rdep_buf0():
        try:
            feature = c.sentence[max(
                {arc for arc in c.arcs if arc.h == c.buffer[0]}, key=lambda arc: arc.d).d].deprel
        except ValueError:
            feature = 'None'
        return feature

    def deprel_ldep_stk0():
        try:
            feature = c.sentence[min(
                {arc for arc in c.arcs if arc.h == c.stack[-1]}, key=lambda arc: arc.d).d].deprel
        except ValueError:
            feature = 'None'
        except IndexError:
            feature = 'None'
        return feature

    def deprel_ldep_buf0():
        try:
            feature = c.sentence[min(
                {arc for arc in c.arcs if arc.h == c.buffer[0]}, key=lambda arc: arc.d).d].deprel
        except ValueError:
            feature = 'None'
        return feature


    # todo put them into the vector. check if it breaks

    return ['bias',  # bias what?
            'b0.form=' + form_of_buffer_front(), 'b0.pos=' + pos_of_buffer_front(),
            's0.form=' + form_of_stack_top(), 's0.pos=' + pos_of_stack_top(),
            'b1.pos=' + pos_of_second_buffer_item(), 's1.pos=' + pos_of_second_stack_item(),
            'ld(b0).pos=' + pos_of_leftmost_dep_of_buffer_front(),
            's0.pos b0.pos=' + pos_of_stack_top_and_pos_of_buffer_front(),
            's0.pos b0.form=' + pos_of_stack_top_and_form_of_buffer_front(),
            's0.form b0.pos=' + form_of_stack_top_and_pos_of_buffer_front(),
            's0.form b0.form=' + form_of_stack_top_and_form_of_buffer_front(),
            's0.lemma=' + lemma_stk0(),  # new features start
            'b0.lemma=' + lemma_buf0(),
            'b1.form=' + form_buf1(),
            'b2.pos=' + pos_buf2(),
            'b3.pos=' + pos_buf3(),
            'rd(s0).deprel=' + deprel_rdep_stk0(),
            'ld(s0).deprel=' + deprel_ldep_stk0(),
            'rd(b0).deprel=' + deprel_rdep_buf0(),
            'ld(b0).deprel=' + deprel_ldep_buf0()
            ]

# fixme perhaps a better test case