def no_progress_loss(iteration_stop_count=20, percent_increase=0.0):
    """
    Stop function that will stop after X iteration if the loss doesn't increase
    Parameters
    ----------
    iteration_stop_count: int
        search will stop if the loss doesn't improve after this number of iteration
    percent_increase: float
        allow this percentage of variation within iteration_stop_count.
        Early stop will be triggered if the data didn't change for more than this number
        after iteration_stop_count rounds
    """

    def stop_fn(trials, best_loss=None, iteration_no_progress=0):
        new_loss = trials.trials[len(trials.trials) - 1]["result"]["loss"]
        if best_loss is None:
            return False, [new_loss, iteration_no_progress + 1]
        best_loss_threshold = best_loss - abs(best_loss * (percent_increase / 100.0))
        if new_loss < best_loss_threshold:
            best_loss = new_loss
            iteration_no_progress = 0
        else:
            iteration_no_progress += 1

        return (
            iteration_no_progress >= iteration_stop_count,
            [best_loss, iteration_no_progress],
        )

    return stop_fn

def no_progress_loss_1(iteration_stop_count=5, percent_increase=0.0):
    """
    Stop function that will stop after X iteration if the loss doesn't increase
    Parameters
    ----------
    iteration_stop_count: int
        search will stop if the loss doesn't improve after this number of iteration
    percent_increase: float
        allow this percentage of variation within iteration_stop_count.
        Early stop will be triggered if the data didn't change for more than this number
        after iteration_stop_count rounds
    """

    def stop_fn(trials, best_losses=[]):
        new_loss = trials.trials[len(trials.trials) - 1]["result"]["loss"]
        if not best_losses:
            return False, [new_loss]
        
        best_losses.append(min(new_loss,best_losses[-1])) 

        if len(best_losses)<=iteration_stop_count+1:
            return False, [best_losses]
        else:
            return best_losses[-iteration_stop_count:] == [best_losses[-iteration_stop_count]] * iteration_stop_count, [best_losses]

    return stop_fn