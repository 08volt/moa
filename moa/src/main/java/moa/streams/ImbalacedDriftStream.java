package moa.streams;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

public class ImbalacedDriftStream extends ImbalancedStream{

    public IntOption positionOption = new IntOption("position",
            'p', "Central position of concept drift change.", 0);

    public IntOption widthOption = new IntOption("width",
            'w', "Width of concept drift change.", 1000);

    public ClassOption driftstreamOption = new ClassOption("driftstream", 'd',
            "Imbalance drift Stream.", ExampleStream.class,
            "ImbalancedStream");

    protected int numberInstanceStream;
    private ExampleStream driftStream;


    @Override
    public void prepareForUse(TaskMonitor monitor, ObjectRepository repository) {
        super.prepareForUse(monitor, repository);
        this.driftStream = (ExampleStream) getPreparedClassOption(this.driftstreamOption);
    }

    @Override
    public Example nextInstance() {

        numberInstanceStream++;
        double x = -4.0 * (double) (numberInstanceStream - this.positionOption.getValue()) / (double) this.widthOption.getValue();
        double probabilityDrift = 1.0 / (1.0 + Math.exp(x));
        if (this.random.nextDouble() > probabilityDrift)
            return super.nextInstance();

        return this.driftStream.nextInstance();


    }
}
